"""
Partial-convolution UNet with a self-attention (non-local) block at the bottleneck.

Drop-in replacement for Daniel's UnetSolverBilin. Two ideas, aimed at sparse SLA:

1. PARTIAL CONVOLUTIONS in the encoder (Liu et al. 2018 / Uhrig et al. 2017):
   each conv is renormalized by the fraction of *valid* pixels in its receptive
   field, and the validity mask is propagated so holes shrink layer by layer.
   This fixes the `nan_to_num -> 0` lie (a gap is no longer treated as a real
   "0 anomaly" measurement) and lets information spread out of the tracks.

2. SELF-ATTENTION at the bottleneck: connects any two coarse-grid locations in a
   single layer, giving a global receptive field so distant observations can
   inform a gap (the O(sqrt(depth)) effective-receptive-field limit of plain
   convs is the thing this defeats).

The mask is derived inside `forward` from `isfinite(batch.input)`, so nothing
outside this file changes: same datamodule, same loss, same Lit module, same
ensemble inference. `nan_to_num` stays (needed so masked convs don't see NaN),
but the mask is now taken BEFORE it.

Put at contrib/glorys12/punet.py and point solver._target_ at PartialUNetSolver.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c, prefer=32):
    for g in (prefer, 16, 8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


def _pad16(x, m, mult=16):
    # 4 downsamples -> sizes must be multiples of 16 (720, 1440, 256, 512 already are)
    _, _, H, W = x.shape
    ph, pw = (mult - H % mult) % mult, (mult - W % mult) % mult
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
        m = F.pad(m, (0, pw, 0, ph))          # padded region = invalid
    return x, m, (ph, pw)


def _unpad(x, pads):
    ph, pw = pads
    if ph or pw:
        x = x[..., : x.shape[-2] - ph, : x.shape[-1] - pw]
    return x


class PartialConv2d(nn.Module):
    """Partial conv. Accepts a per-channel mask (first layer) OR a single-channel
    mask (all later layers). Always returns a single-channel updated mask."""
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, padding, bias=bias)
        self.in_ch, self.k, self.stride, self.padding = in_ch, k, stride, padding
        self.register_buffer("k_full", torch.ones(1, in_ch, k, k))
        self.register_buffer("k_one", torch.ones(1, 1, k, k))

    def forward(self, x, m):
        with torch.no_grad():
            if m.shape[1] == 1:                                  # shared mask
                um = F.conv2d(m, self.k_one, stride=self.stride, padding=self.padding)
                slide = self.k * self.k
            else:                                                # per-channel (stem)
                um = F.conv2d(m, self.k_full, stride=self.stride, padding=self.padding)
                slide = self.in_ch * self.k * self.k
            ratio = slide / (um + 1e-8)
            um = torch.clamp(um, 0, 1)
            ratio = ratio * um
        out = self.conv(x * m)
        if self.conv.bias is not None:
            b = self.conv.bias.view(1, -1, 1, 1)
            out = (out - b) * ratio + b
        else:
            out = out * ratio
        return out * um, um


class PConvResBlock(nn.Module):
    """Pre-activation residual block built from partial convs (encoder/bottleneck)."""
    def __init__(self, cin, cout, dropout=0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(_gn(cin), cin)
        self.act = nn.SiLU()
        self.pc1 = PartialConv2d(cin, cout, 3, 1, 1)
        self.n2 = nn.GroupNorm(_gn(cout), cout)
        self.drop = nn.Dropout(dropout)
        self.pc2 = PartialConv2d(cout, cout, 3, 1, 1)
        self.skip = PartialConv2d(cin, cout, 1, 1, 0) if cin != cout else None

    def forward(self, x, m):
        h = self.act(self.n1(x))
        h, m1 = self.pc1(h, m)
        h = self.drop(self.act(self.n2(h)))
        h, m2 = self.pc2(h, m1)
        xs = self.skip(x, m)[0] if self.skip is not None else x
        return h + xs, m2


class SelfAttention2d(nn.Module):
    """Multi-head self-attention over spatial tokens (the 'non-local' block).
    Applied only at the /16 bottleneck (small token count)."""
    def __init__(self, ch, heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.GroupNorm(_gn(ch), ch)
        self.mha = nn.MultiheadAttention(ch, heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        t = self.norm(x).flatten(2).transpose(1, 2)          # (B, H*W, C)
        o, _ = self.mha(t, t, t, need_weights=False)
        return x + o.transpose(1, 2).reshape(b, c, h, w)


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class ResBlock(nn.Module):
    """Plain residual block (decoder — the field is dense by here, no mask needed)."""
    def __init__(self, cin, cout, dropout=0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(_gn(cin), cin)
        self.act = nn.SiLU()
        self.c1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.n2 = nn.GroupNorm(_gn(cout), cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(self.act(self.n1(x)))
        h = self.c2(self.drop(self.act(self.n2(h))))
        return h + self.skip(x)


class PartialUNetSolver(nn.Module):
    def __init__(self, dim_in=15, dim_out=None, base=48, attn_heads=8, dropout=0.1, **_ignored):
        super().__init__()
        self._dbg = 0                      # <-- add this to count batch number
        dim_out = dim_out or dim_in
        self.dim_in, self.dim_out = dim_in, dim_out
        c1, c2, c3, c4 = base, 2 * base, 4 * base, 8 * base

        self.stem = PartialConv2d(dim_in, c1, 3, 1, 1)          # per-channel mask enters here
        self.enc1, self.down1 = PConvResBlock(c1, c1, dropout), PartialConv2d(c1, c2, 3, 2, 1)
        self.enc2, self.down2 = PConvResBlock(c2, c2, dropout), PartialConv2d(c2, c3, 3, 2, 1)
        self.enc3, self.down3 = PConvResBlock(c3, c3, dropout), PartialConv2d(c3, c4, 3, 2, 1)
        self.enc4, self.down4 = PConvResBlock(c4, c4, dropout), PartialConv2d(c4, c4, 3, 2, 1)

        self.bott1 = PConvResBlock(c4, c4, dropout)
        self.attn = SelfAttention2d(c4, attn_heads, dropout)    # global connectivity at /16
        self.bott2 = PConvResBlock(c4, c4, dropout)

        self.up4, self.dec4 = Up(c4, c4), ResBlock(c4 + c4, c4, dropout)
        self.up3, self.dec3 = Up(c4, c3), ResBlock(c3 + c3, c3, dropout)
        self.up2, self.dec2 = Up(c3, c2), ResBlock(c2 + c2, c2, dropout)
        self.up1, self.dec1 = Up(c2, c1), ResBlock(c1 + c1, c1, dropout)

        self.head = nn.Sequential(
            nn.GroupNorm(_gn(c1), c1), nn.SiLU(), nn.Conv2d(c1, dim_out, 3, 1, 1)
        )

        n = sum(p.numel() for p in self.parameters())
        print(f"[PartialUNetSolver] base={base} params={n/1e6:.2f}M "
              f"(dim_in={dim_in}, dim_out={dim_out}, heads={attn_heads})")

    def forward(self, batch):
        x = batch.input                       # (B, dim_in, H, W), NaN in gaps
        m = torch.isfinite(x).float()         # validity mask, taken BEFORE nan_to_num
        
        if self._dbg < 5:                  # only the first call
            print(f"[punet] mask coverage = {m.mean().item():.4f}  shape={tuple(m.shape)}", flush=True)
            self._dbg += 1
        
        x = x.nan_to_num()                    # NaN -> 0 (masked out inside the pconvs)
        x, m, pads = _pad16(x, m)

        x, m = self.stem(x, m)
        x, m = self.enc1(x, m); s1 = x; x, m = self.down1(x, m)
        x, m = self.enc2(x, m); s2 = x; x, m = self.down2(x, m)
        x, m = self.enc3(x, m); s3 = x; x, m = self.down3(x, m)
        x, m = self.enc4(x, m); s4 = x; x, m = self.down4(x, m)

        x, m = self.bott1(x, m)
        x = self.attn(x)
        x, m = self.bott2(x, m)

        x = self.dec4(torch.cat([self.up4(x), s4], 1))
        x = self.dec3(torch.cat([self.up3(x), s3], 1))
        x = self.dec2(torch.cat([self.up2(x), s2], 1))
        x = self.dec1(torch.cat([self.up1(x), s1], 1))

        return _unpad(self.head(x), pads)