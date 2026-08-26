"""
Adapter: use the 3D FiLM/temporal-attention UNet as a drop-in solver in place of
Daniel's UnetSolverBilin.

Shapes:
    batch.input : (B, T, H, W)   T = patch_dims.time (time-as-channels, 4D)
    UNet3D_Transformer_priorcost.predict() already does
        x.unsqueeze(1) -> (B, 1, T, H, W)   [line ~472 of unet3d.py]
        out.squeeze(1) -> (B, T, H, W)      [line ~528]
    so NO unsqueeze/squeeze is needed here; only the batch -> tensor adaptation.

Save the fetched file as  contrib/glorys12/unet3d.py  and this as
contrib/glorys12/unet3d_solver.py, then point solver._target_ at UNet3DSolver.
"""
import torch
import torch.nn as nn

from .unet3d import UNet3D_Transformer_priorcost


class UNet3DSolver(nn.Module):
    """3D UNet (2 ResBlocks/stage, FiLM, temporal attention at the bottleneck)
    wrapped to Daniel's solver API: forward(batch) -> (B, T, H, W)."""

    def __init__(
        self,
        channels=(48, 96, 192, 384),
        embed_dim=256,
        dropout=0.1,
        attn_heads=4,
        use_mask_channel=False,   # feed the validity mask as a 2nd channel (see note)
        **_ignored               # absorbs channel_dims/bias/... merged from unet.yaml
    ):
        super().__init__()
        self.use_mask_channel = use_mask_channel
        self.net = UNet3D_Transformer_priorcost(
            in_channels=2 if use_mask_channel else 1,
            channels=tuple(channels),
            embed_dim=embed_dim,
            dropout=dropout,
            attn_heads=attn_heads,
        )
        n = sum(p.numel() for p in self.parameters())
        print(f"[UNet3DSolver] channels={tuple(channels)} heads={attn_heads} "
              f"mask_ch={use_mask_channel} params={n/1e6:.2f}M", flush=True)

    def forward(self, batch):
        x = batch.input                       # (B, T, H, W), NaN in gaps
        if self.use_mask_channel:
            m = torch.isfinite(x).float().unsqueeze(1)     # (B, 1, T, H, W)
            x = x.nan_to_num()
            # predict() unsqueezes x itself; y is concatenated as an extra channel
            t = torch.zeros((x.shape[0],), device=x.device)
            return self.net.predict(x, timesteps=t, y=m)
        x = x.nan_to_num()
        return self.net(x, timesteps=None)    # (B, T, H, W)