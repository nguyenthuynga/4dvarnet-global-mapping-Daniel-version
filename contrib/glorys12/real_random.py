"""
Real-data (OSE) training with random satellite subsetting.

Per patch:
  - pick k of the pool satellites at random  ->  union their (gappy) tracks = INPUT
  - the TARGET is the union of ALL pool satellites
The held-out satellites (not picked) carry the supervised signal, so the network
learns to RECONSTRUCT unseen tracks instead of copying the input.

Real along-track data is gappy, which forces three changes vs the GLORYS/OSSE setup:
  1. reconstruction loss is computed only on FINITE target points
     (ocean4dvarnet.weighted_mse already masks to finite entries -> NaN-safe);
  2. the spatial-gradient (sobel) loss CANNOT work on a sparse target
     (sobel(tgt) is NaN almost everywhere) -> set w_grad_mse: 0 in the config;
  3. the batch-reject threshold must be lowered: a track-only target is never
     ~50% finite, so the stock `< 0.5` test would reject every batch.

Put this at  contrib/glorys12/real_random.py  and point a new xp config at it.
"""
import functools as ft
import warnings

import numpy as np
import torch
import xarray as xr

from ocean4dvarnet.data import BaseDataModule

from .unet import (
    TrainingItemwithLonLat,
    LitUnetFromLit4dVarNetIgnoreNaN,
    NormParamsNotProvided,
)


# ----------------------------------------------------------------------
# loader: open the per-satellite gridded files lazily, stack on a 'sat' axis
# ----------------------------------------------------------------------
def load_real_satellites(sat_paths, var="sla", **_ignored):
    # _ignored absorbs leftover tgt_path/inp_path/tgt_var/inp_var from the
    # merged base unet.yaml so Hydra's deep-merge can't crash the call
    
    das = []
    for p in sat_paths:
        ds = xr.open_dataset(p)              # lazy; only patches get read later
        ren = {}
        if "latitude" in ds.dims:
            ren["latitude"] = "lat"
        if "longitude" in ds.dims:
            ren["longitude"] = "lon"
        das.append(ds.rename(ren)[var])
    sats = xr.concat(das, dim="sat")         # (sat, time, lat, lon), gappy (NaN off-track)
    print(f"[real-random] pool = {len(das)} satellites, dims = {dict(sats.sizes)}")
    assert "tgt_path" not in _ignored or sat_paths, "sat_paths missing — input_da merge problem"
    return sats


class RealRandomDataModule(BaseDataModule):
    def __init__(self, *args, k_min=1, k_max=None, **kwargs):
        self.k_min = k_min
        self.k_max = k_max
        super().__init__(*args, **kwargs)
        self.sats = self.input_da            # the (sat, time, lat, lon) DataArray

    def norm_stats(self):
        if self._norm_stats is None:
            raise NormParamsNotProvided()
        return self._norm_stats

    def post_fn(self, phase):
        m, s = self.norm_stats()[phase]

        def normalize(x):
            return (x - m) / s

        # normalize input & target only (lon/lat carried untouched)
        return ft.partial(
            ft.reduce,
            lambda i, f: f(i),
            [
                TrainingItemwithLonLat._make,
                lambda it: it._replace(tgt=normalize(it.tgt)),
                lambda it: it._replace(input=normalize(it.input)),
            ],
        )

    def setup(self, stage="test"):
        self.train_ds = RealRandomDataset(
            self.sats.sel(self.domains["train"]),
            **self.xrds_kw["train"],
            postpro_fn=self.post_fn("train"),
            k_min=self.k_min, k_max=self.k_max,
        )
        self.val_ds = RealRandomDataset(
            self.sats.sel(self.domains["val"]),
            **self.xrds_kw["val"],
            postpro_fn=self.post_fn("val"),
            k_min=self.k_min, k_max=self.k_max,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, shuffle=False, batch_size=1, num_workers=1,
        )


class RealRandomDataset(torch.utils.data.Dataset):
    def __init__(self, sats, patch_dims, domain_limits=None, strides=None,
                 postpro_fn=None, noise=None, noise_type=None,
                 k_min=1, k_max=None, **kwargs):
        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.sats = sats.sel(**(domain_limits or {}))
        self.n_sat = self.sats.sizes["sat"]
        self.k_min = k_min
        self.k_max = k_max if k_max is not None else self.n_sat - 1   # >=1 held out
        assert 1 <= self.k_min <= self.k_max <= self.n_sat, "bad k range"

        self.patch_dims = patch_dims
        self.strides = strides or {}
        spatial_dims = [d for d in self.sats.dims if d != "sat"]      # time, lat, lon
        ds_dims = {d: self.sats.sizes[d] for d in spatial_dims}
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

    def __getitem__(self, item):
        sl = {}
        for dim, idx in zip(self.ds_size.keys(),
                            np.unravel_index(item, tuple(self.ds_size.values()))):
            sl[dim] = slice(self.strides.get(dim, 1) * idx,
                            self.strides.get(dim, 1) * idx + self.patch_dims[dim])

        _ds = self.sats.isel(**sl)            # (sat, time, lat<=patch, lon<=patch)

        # longitude periodic extension (sat dim rides along untouched)
        offset_lon = 0
        if self.periodic_dim and len(_ds.lon) < self.patch_dims["lon"]:
            offset_lon = self.patch_dims["lon"] - len(_ds.lon)
            sl["lon"] = slice(sl["lon"].start - offset_lon, sl["lon"].stop - offset_lon)
            _ds = (self.sats.isel(time=sl["time"]).roll(lon=-offset_lon)
                   .assign_coords(lon=lambda x: x.lon - offset_lon).sortby("lon")
                   .isel({d: v for d, v in sl.items() if d != "time"}))

        # latitude padding
        offset_lat = 0
        if self.padded_dim and len(_ds.lat) < self.patch_dims["lat"]:
            offset_lat = self.patch_dims["lat"] - len(_ds.lat)
            _ds = _ds.pad(pad_width=dict(lat=(0, offset_lat)), mode="constant", constant_values=np.nan)

        if self.return_coords:
            return _ds.isel(sat=0).coords.to_dataset()[list(self.patch_dims)]

        stack = _ds.data.astype(np.float32)   # (sat, T, H, W) -- reads this patch only

        # input = union of a random k-subset ; target = union of ALL satellites
        k = int(self._rng.integers(self.k_min, self.k_max + 1))
        idx = self._rng.choice(self.n_sat, size=k, replace=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN cells
            tgt = np.nanmean(stack, axis=0)
            inp = np.nanmean(stack[idx], axis=0)
        

        if self.noise is not None:
            if self.noise_type == "uniform-constant":
                inp = inp + self._rng.uniform(-self.noise, self.noise, inp.shape).astype(np.float32)
            elif self.noise_type == "gaussian+uniform":
                scale = self._rng.uniform(0.0, self.noise, 1).astype(np.float32)
                inp = inp + scale * self._rng.normal(0.0, 1.0, inp.shape).astype(np.float32)

        item = TrainingItemwithLonLat(
            inp.astype(np.float32),
            tgt.astype(np.float32),
            _ds.coords["lon"].data.astype(np.float32),
            _ds.coords["lat"].data.astype(np.float32),
        )
        return self.postpro_fn(item) if self.postpro_fn is not None else item


# ----------------------------------------------------------------------
# Lit module: tolerant of sparse (real) targets
#   - lowered reject threshold (a track-only target is never ~50% finite)
#   - keep w_grad_mse: 0 in the config (sobel loss is meaningless on sparse tgt)
# ----------------------------------------------------------------------
class LitUnetReal(LitUnetFromLit4dVarNetIgnoreNaN):
    min_tgt_frac = 0.02 #at least 2 percent is required #1e-2

    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < self.min_tgt_frac:
            return None, None
        _, out = self.base_step(batch, phase)
        loss_mse = self.loss_mse(batch, out, phase)
        training_loss = self.w_mse * loss_mse[0] + self.w_grad_mse * loss_mse[1]
        self.log(f"{phase}_gloss", loss_mse[1], prog_bar=False, on_step=False, on_epoch=True)
        return training_loss, out