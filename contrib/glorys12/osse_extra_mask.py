"""
Extra observation masking on top of a FIXED constellation (OSSE).

Motivated by Dorffer et al. (FishPredict), Fig. 11 / Table III: instead of
varying which satellites are present, keep the constellation fixed and remove
observations either as random rectangular patches or as random individual
pixels.  In that paper the patch-based removal is the best-performing scheme
and acts as data augmentation.

    extra_mask: null       -> nothing removed (same as RealRandomOSSEDataset)
    extra_mask: squares    -> blank random rectangles until keep_frac remains
    extra_mask: points     -> drop each observed pixel with prob 1-keep_frac

Adapted from `rand_obs()` in 4dvarnet-starter/src/data.py, with two changes
that matter here:

  1. the reference gates removal on `sum(obs) > .25*dlat*dlon`, i.e. it assumes
     a mostly-dense field.  A 7-nadir+SWOT union covers ~8% of the patch, so
     that gate would NEVER fire and nothing would be removed.  Here the removal
     target is a fraction of the *observed* pixels instead.
  2. the reference writes 0 for removed pixels.  This pipeline marks gaps with
     NaN (the solver does nan_to_num, the Lit module counts isfinite), so
     writing 0 would turn every gap into a real "0 m anomaly" measurement.
     Removed pixels are set to NaN.

Put at  contrib/glorys12/osse_extra_mask.py
"""
import numpy as np
import torch

from .real_random_osse import RealRandomOSSEDataModule, RealRandomOSSEDataset


# ---------------------------------------------------------------- removal ---
def remove_points(inp, keep_frac, rng):
    """Drop each observed pixel independently with probability 1-keep_frac."""
    out = np.array(inp, copy=True)
    obs = np.isfinite(out)
    drop = obs & (rng.random(out.shape) > keep_frac)
    out[drop] = np.nan
    return out


def remove_squares(inp, keep_frac, half_min, half_max, rng, max_iter=2000):
    """Blank random rectangles, per time step, until at most keep_frac of the
    originally observed pixels of that time step remain."""
    out = np.array(inp, copy=True)
    T, H, W = out.shape
    for t in range(T):
        n0 = int(np.isfinite(out[t]).sum())
        if n0 == 0:
            continue
        target = keep_frac * n0
        it = 0
        while int(np.isfinite(out[t]).sum()) > target and it < max_iter:
            hh = int(rng.integers(half_min, half_max))
            hw = int(rng.integers(half_min, half_max))
            i = int(rng.integers(0, H))
            j = int(rng.integers(0, W))
            out[t,
                max(0, i - hh):min(H, i + hh + 1),
                max(0, j - hw):min(W, j + hw + 1)] = np.nan
            it += 1
    return out


# ---------------------------------------------------------------- dataset ---
class OSSEExtraMaskDataset(RealRandomOSSEDataset):
    """RealRandomOSSEDataset + an extra removal pass on the INPUT only."""

    def __init__(self, *args, extra_mask=None, keep_frac=0.5,
                 square_half_min=2, square_half_max=10,
                 debug_items=0, tag="", **kwargs):
        super().__init__(*args, **kwargs)
        if extra_mask not in (None, "none", "squares", "points"):
            raise ValueError(
                f"extra_mask must be null, 'squares' or 'points', got {extra_mask!r}")
        self.extra_mask = None if extra_mask == "none" else extra_mask
        self.keep_frac = keep_frac
        self.square_half_min = square_half_min
        self.square_half_max = square_half_max
        self.debug_items = debug_items
        self.tag = tag
        self._dbg = 0
        # (before, after) per item, for check_extra_mask.py (num_workers=0 only)
        self.counts = []

    def __getitem__(self, item):
        out = super().__getitem__(item)

        # the coords path returns an xarray Dataset, not a TrainingItem
        if self.return_coords or self.extra_mask is None:
            return out

        inp = np.asarray(out.input)
        before = int(np.isfinite(inp).sum())

        if self.extra_mask == "squares":
            inp = remove_squares(inp, self.keep_frac,
                                 self.square_half_min, self.square_half_max,
                                 self._rng)
        else:
            inp = remove_points(inp, self.keep_frac, self._rng)

        after = int(np.isfinite(inp).sum())
        self.counts.append((before, after))

        if self._dbg < self.debug_items:
            wi = torch.utils.data.get_worker_info()
            who = f"worker{wi.id}" if wi is not None else "main"
            tot = inp.size
            kept = 100.0 * after / max(before, 1)
            print(f"[extra-mask {self.tag}/{self.extra_mask}|{who}] item {self._dbg}: "
                  f"observed {before:>8d} -> {after:>8d}   "
                  f"({100.0*before/tot:5.2f}% -> {100.0*after/tot:5.2f}% of patch, "
                  f"kept {kept:5.1f}%)", flush=True)
            self._dbg += 1

        return out._replace(input=inp.astype(np.float32))


# ------------------------------------------------------------- datamodule ---
class OSSEExtraMaskDataModule(RealRandomOSSEDataModule):
    def __init__(self, *args, extra_mask=None, keep_frac=0.5,
                 square_half_min=2, square_half_max=10,
                 apply_to_val=True, debug_items=0, **kwargs):
        self.extra_kw = dict(
            extra_mask=extra_mask,
            keep_frac=keep_frac,
            square_half_min=square_half_min,
            square_half_max=square_half_max,
            debug_items=debug_items,
        )
        self.apply_to_val = apply_to_val
        super().__init__(*args, **kwargs)
        print(f"[extra-mask] mode={extra_mask} keep_frac={keep_frac} "
              f"squares_half=[{square_half_min},{square_half_max}) "
              f"apply_to_val={apply_to_val}", flush=True)

    def setup(self, stage="test"):
        self.train_ds = OSSEExtraMaskDataset(
            self.tgt.sel(self.domains["train"]),
            self.masks, self.target_idx,
            **self.xrds_kw["train"],
            postpro_fn=self.post_fn("train"),
            mode_target=self.mode_target,
            k_min=self.k_min, k_max=self.k_max,
            tag="train", **self.extra_kw,
        )

        val_kw = dict(self.extra_kw)
        if not self.apply_to_val:
            val_kw["extra_mask"] = None
        self.val_ds = OSSEExtraMaskDataset(
            self.tgt.sel(self.domains["val"]),
            self.masks, self.target_idx,
            **self.xrds_kw["val"],
            postpro_fn=self.post_fn("val"),
            mode_target=self.mode_target,
            k_min=self.k_min, k_max=self.k_max,
            tag="val", **val_kw,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, shuffle=False, batch_size=1, num_workers=1,
        )