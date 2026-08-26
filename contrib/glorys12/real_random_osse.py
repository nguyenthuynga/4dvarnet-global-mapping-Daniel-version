"""
OSSE pretrain that MATCHES the OSE random-k finetune geometry.

Same GLORYS field, two footprints per patch:
  - TARGET = GLORYS masked by the FULL union (7 nadirs + SWOT)      [fixed]
  - INPUT  = GLORYS masked by a RANDOM-k union of the pool masks    [random]

So the input->target task is identical to the OSE run (union-of-k -> union-of-all);
only the data source changes (GLORYS here, real tracks in the finetune). The plain
UNet (dim_in=15) is kept so the weights transfer directly into real_random.py.

Put at  contrib/glorys12/real_random_osse.py  and point an xp config at it.
Masks are the 2024 365-day footprints, tiled onto the GLORYS years with `% 365`.
"""
import functools as ft

import numpy as np
import torch
import xarray as xr

from ocean4dvarnet.data import BaseDataModule, TrainingItem

from .unet import NormParamsNotProvided


def load_glorys12_osse_masks(tgt_path, mask_paths, target_mask_idx=None,
                             tgt_var="sla", mask_var="mask", **_ignored):
    """Return (tgt, masks, target_idx): GLORYS + a list of per-satellite boolean masks.
    target_mask_idx = which masks form the TARGET union (default: all of them)."""
    def _open(path, var):
        ds = xr.open_dataset(path)
        ren = {}
        if "latitude" in ds.dims:
            ren["latitude"] = "lat"
        if "longitude" in ds.dims:
            ren["longitude"] = "lon"
        return ds.rename(ren)[var]

    tgt = _open(tgt_path, tgt_var)
    masks = [np.isfinite(_open(p, mask_var)) for p in mask_paths]
    target_idx = list(range(len(masks))) if target_mask_idx is None else list(target_mask_idx)
    for i, m in enumerate(masks):
        # tag = "TARGET" if i in target_idx else "input-only"
        print(f"[osse-random] mask {i} coverage={float(m.mean()):.4f}") #  ({tag})
    print(f"[osse-random] {len(masks)} masks; target union = idx {target_idx}")
    return tgt, masks, target_idx


class RealRandomOSSEDataModule(BaseDataModule):
    def __init__(self, *args, mode_target=None, k_min=1, k_max=None, **kwargs):
        self.mode_target = mode_target
        self.k_min = k_min
        self.k_max = k_max
        super().__init__(*args, **kwargs)
        self.tgt, self.masks, self.target_idx = self.input_da

    def norm_stats(self):
        if self._norm_stats is None:
            raise NormParamsNotProvided()
        return self._norm_stats

    def post_fn(self, phase):
        m, s = self.norm_stats()[phase]

        def normalize(x):
            return (x - m) / s

        return ft.partial(
            ft.reduce,
            lambda i, f: f(i),
            [
                TrainingItem._make,
                lambda it: it._replace(tgt=normalize(it.tgt)),
                lambda it: it._replace(input=normalize(it.input)),
            ],
        )

    def setup(self, stage="test"):
        self.train_ds = RealRandomOSSEDataset(
            self.tgt.sel(self.domains["train"]),
            self.masks, self.target_idx,
            **self.xrds_kw["train"], postpro_fn=self.post_fn("train"),
            mode_target=self.mode_target,
            k_min=self.k_min, k_max=self.k_max,
        )
        self.val_ds = RealRandomOSSEDataset(
            self.tgt.sel(self.domains["val"]),
            self.masks, self.target_idx,
            **self.xrds_kw["val"], postpro_fn=self.post_fn("val"),
            mode_target=self.mode_target,
            k_min=self.k_min, k_max=self.k_max,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, shuffle=False, batch_size=1, num_workers=1,
        )


class RealRandomOSSEDataset(torch.utils.data.Dataset):
    def __init__(self, ds, masks, target_idx, patch_dims, mode_target, domain_limits=None, strides=None,
                 postpro_fn=None, noise=None, noise_type=None,
                 k_min=1, k_max=None, **kwargs):
        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.ds = ds.sel(**(domain_limits or {}))

        # crop each mask to the spatial domain (keep its 365-day time axis)
        self.masks = []
        for m in masks:
            if domain_limits is not None:
                spatial = {k: v for k, v in domain_limits.items()
                           if k in m.dims and k != "time"}
                m = m.sel(**spatial)
            self.masks.append(m)
        self.n_sat = len(self.masks)
        self.target_idx = list(target_idx)

        self.mode_target = mode_target
        self.k_min = k_min
        self.k_max = k_max if k_max is not None else self.n_sat
        assert 1 <= self.k_min <= self.k_max <= self.n_sat, "bad k range"

        self.patch_dims = patch_dims
        self.strides = strides or {}
        _dims = ("variable",) + tuple(k for k in self.ds.dims)
        _shape = (2,) + tuple(self.ds[k].shape[0] for k in self.ds.dims)
        ds_dims = dict(zip(_dims, _shape))
        self.ds_size = {
            dim: max((ds_dims[dim] - patch_dims[dim]) // strides.get(dim, 1) + 1, 0)
            for dim in patch_dims
        }
        self.periodic_dim = kwargs.get("periodic_dim")
        if self.periodic_dim and (ds_dims[self.periodic_dim] - patch_dims[self.periodic_dim]) % strides[self.periodic_dim] != 0:
            self.ds_size[self.periodic_dim] += 1
        self.padded_dim = kwargs.get("padded_dim")
        if self.padded_dim and (ds_dims[self.padded_dim] - patch_dims[self.padded_dim]) % strides[self.padded_dim] != 0:
            self.ds_size[self.padded_dim] += 1

        self._rng = np.random.default_rng()
        self.noise = noise
        self.noise_type = noise_type or "uniform-constant"

    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_coords(self):
        self.return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self.return_coords = False
            return coords

    def _union_mask(self, idx, sl_mask, offset_lon, offset_lat):
        """Union (logical OR) of the chosen satellites' footprints for this patch."""
        acc = None
        for j in idx:
            mj = self.masks[j].isel(time=sl_mask["time"])
            if offset_lon:
                mj = (mj.roll(lon=-offset_lon)
                      .assign_coords(lon=lambda x: x.lon - offset_lon).sortby("lon"))
            mj = mj.isel({d: v for d, v in sl_mask.items() if d != "time"})
            if offset_lat:
                mj = mj.pad(pad_width=dict(lat=(0, offset_lat)),
                            mode="constant", constant_values=False)
            vals = np.asarray(mj.values, dtype=bool)
            acc = vals if acc is None else (acc | vals)
        return acc

    def __getitem__(self, item):
        sl = {}
        for dim, idx in zip(self.ds_size.keys(),
                            np.unravel_index(item, tuple(self.ds_size.values()))):
            sl[dim] = slice(self.strides.get(dim, 1) * idx,
                            self.strides.get(dim, 1) * idx + self.patch_dims[dim])

        _ds = self.ds.isel(**sl)

        # ---- draw the input count k + random satellite subset ----
        k = int(self._rng.integers(self.k_min, self.k_max + 1))
        sat_idx = self._rng.choice(self.n_sat, size=k, replace=False)

        # longitude periodic extension (mirror onto the target)
        offset_lon = 0
        if self.periodic_dim and len(_ds.lon) < self.patch_dims["lon"]:
            offset_lon = self.patch_dims["lon"] - len(_ds.lon)
            sl["lon"] = slice(sl["lon"].start - offset_lon, sl["lon"].stop - offset_lon)
            _ds = (self.ds.isel(time=sl["time"]).roll(lon=-offset_lon)
                   .assign_coords(lon=lambda x: x.lon - offset_lon).sortby("lon")
                   .isel({d: v for d, v in sl.items() if d != "time"}))

        # mask time index: % 365 circular tiling
        start, stop = sl["time"].start % 365, sl["time"].stop % 365
        if start > stop:
            start -= stop
            stop = None
        sl_mask = sl.copy()
        sl_mask["time"] = slice(start, stop)

        # latitude padding (target)
        offset_lat = 0
        if self.padded_dim and len(_ds.lat) < self.patch_dims["lat"]:
            offset_lat = self.patch_dims["lat"] - len(_ds.lat)
            _ds = _ds.pad(pad_width=dict(lat=(0, offset_lat)), mode="constant", constant_values=np.nan)

        if self.return_coords:
            return _ds.coords.to_dataset()[list(self.patch_dims)]

        if self.mode_target=="gappy":
            # two footprints on the SAME GLORYS field
            union_in  = self._union_mask(sat_idx,         sl_mask, offset_lon, offset_lat)  # random-k
            union_tgt = self._union_mask(self.target_idx, sl_mask, offset_lon, offset_lat)  # full union
            tgt = _ds.where(union_tgt).data.astype(np.float32)   # GLORYS on the full-union tracks
            inp = _ds.where(union_in).data.astype(np.float32)    # GLORYS on the random-k tracks
        elif self.mode_target=="gapfree":
            union_in = self._union_mask(sat_idx, sl_mask, offset_lon, offset_lat)
            tgt = _ds.data.astype(np.float32)                    # dense gap-free GLORYS
            inp = _ds.where(union_in).data.astype(np.float32)    # GLORYS on the random-k tracks
        else:
            raise ValueError(f"mode_target must be 'gappy' or 'gapfree', got {self.mode_target!r}")

        if self.noise is not None:
            if self.noise_type == "uniform-constant":
                inp = inp + self._rng.uniform(-self.noise, self.noise, inp.shape).astype(np.float32)
            elif self.noise_type == "gaussian+uniform":
                scale = self._rng.uniform(0.0, self.noise, 1).astype(np.float32)
                inp = inp + scale * self._rng.normal(0.0, 1.0, inp.shape).astype(np.float32)

        item = TrainingItem(inp.astype(np.float32), tgt)
        return self.postpro_fn(item) if self.postpro_fn is not None else item