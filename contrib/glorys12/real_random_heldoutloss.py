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

Put this at  contrib/glorys12/real_random_heldoutloss.py  (next to the original
real_random.py, which stays unchanged) and point an xp config at it.

[HELDOUT] Optional `target_mode` (datamodule argument):
  - "all"     (default, behaviour unchanged): TARGET = union of ALL pool satellites
  - "heldout": TARGET = union of the satellites NOT picked for the input, so the
               loss never compares the network with its own (noisy) input values.
"""
import functools as ft
import warnings

import numpy as np
import kornia.filters as kfilts   # [GRAD 0] same import as unet.py (only for the logged share)
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
    def __init__(self, *args, k_min=1, k_max=None, target_mode="all", **kwargs):
        self.k_min = k_min
        self.k_max = k_max
        self.target_mode = target_mode       # [HELDOUT 1] "all" (default) or "heldout"
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
            target_mode=self.target_mode,    # [HELDOUT 2]
        )
        self.val_ds = RealRandomDataset(
            self.sats.sel(self.domains["val"]),
            **self.xrds_kw["val"],
            postpro_fn=self.post_fn("val"),
            k_min=self.k_min, k_max=self.k_max,
            target_mode=self.target_mode,    # [HELDOUT 2]
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, shuffle=False, batch_size=1, num_workers=1,
        )


class MixedOsseOseDataModule(RealRandomDataModule):
    """[MIX 1] OSE datamodule (same arguments as RealRandomDataModule) plus an OSSE datamodule
    (argument `osse`, e.g. a RealRandomOSSEDataModule). Each training step gets one batch of
    each source, as a dict {"ose": ..., "osse": ...}. Validation is the OSE validation only."""

    def __init__(self, *args, osse=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert osse is not None, "MixedOsseOseDataModule needs an 'osse' datamodule"
        self.osse = osse

    def setup(self, stage="test"):
        super().setup(stage)
        self.osse.setup(stage)

    def train_dataloader(self):
        return {"ose": super().train_dataloader(), "osse": self.osse.train_dataloader()}


class RealRandomDataset(torch.utils.data.Dataset):
    def __init__(self, sats, patch_dims, domain_limits=None, strides=None,
                 postpro_fn=None, noise=None, noise_type=None,
                 k_min=1, k_max=None, target_mode="all", **kwargs):
        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.sats = sats.sel(**(domain_limits or {}))
        self.n_sat = self.sats.sizes["sat"]
        self.k_min = k_min
        self.k_max = k_max if k_max is not None else self.n_sat - 1   # >=1 held out
        assert 1 <= self.k_min <= self.k_max <= self.n_sat, "bad k range"
        # [HELDOUT 3] "all" = old behaviour ; "heldout" = see __getitem__
        assert target_mode in ("all", "heldout"), f"bad target_mode: {target_mode!r}"
        assert target_mode == "all" or self.k_max < self.n_sat, \
            "target_mode='heldout' needs k_max < number of satellites"
        self.target_mode = target_mode

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
            if self.target_mode == "heldout":
                # [HELDOUT 4] target = only the satellites NOT used as input
                held = np.setdiff1d(np.arange(self.n_sat), idx)
                tgt = np.nanmean(stack[held], axis=0)
            else:
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
def pair_grad_mse(out, tgt, weight):
    """[GRAD 1] Gradient loss that also works on gappy targets (nadir tracks).
    For every pair of neighbouring cells (along lon and along lat) where BOTH target
    cells are valid, compares out[j] - out[i] with tgt[j] - tgt[i]. A pair gets the
    smaller rec_weight of its two cells. Same normalisation as weighted_mse (mean of
    the squared weighted errors). Returns (loss, mask of the target cells used)."""
    out, tgt, weight = out.float(), tgt.float(), weight.float()
    sq, n = out.new_zeros(()), out.new_zeros(())
    used = torch.zeros_like(tgt, dtype=torch.bool)
    for dim in (-1, -2):                                   # along lon, then along lat
        size = out.shape[dim]

        def diff(x):
            return x.narrow(dim, 1, size - 1) - x.narrow(dim, 0, size - 1)

        w = torch.minimum(weight.narrow(dim, 1, size - 1), weight.narrow(dim, 0, size - 1))[None]
        err = (diff(out) - diff(tgt)) * w
        ok = err.isfinite() & (w > 0)
        sq = sq + err[ok].pow(2).sum()
        n = n + ok.sum()
        used.narrow(dim, 1, size - 1).logical_or_(ok)
        used.narrow(dim, 0, size - 1).logical_or_(ok)
    return sq / n.clamp(min=1), used


class LitUnetReal(LitUnetFromLit4dVarNetIgnoreNaN):
    min_tgt_frac = 0.02 #at least 2 percent is required #1e-2

    def __init__(self, *args, min_tgt_frac=None, grad_mode="sobel", osse_weight=0.5, osse_grad_mode="sobel",
                 **kwargs):
        # [HELDOUT 5] optional override from the config; None keeps 0.02
        super().__init__(*args, **kwargs)
        if min_tgt_frac is not None:
            self.min_tgt_frac = min_tgt_frac
        # [GRAD 2] gradient term: "sobel" (default, unchanged) or "pairs" (pair_grad_mse)
        assert grad_mode in ("sobel", "pairs"), f"bad grad_mode: {grad_mode!r}"
        self.grad_mode = grad_mode
        # [MIX 2] only used with MixedOsseOseDataModule: share of the OSSE loss in each step,
        # and the gradient term used on the (gap-free) OSSE target
        assert 0.0 <= osse_weight <= 1.0, f"bad osse_weight: {osse_weight}"
        assert osse_grad_mode in ("sobel", "pairs"), f"bad osse_grad_mode: {osse_grad_mode!r}"
        self.osse_weight, self.osse_grad_mode = osse_weight, osse_grad_mode
        self._log_tag = None

    def log(self, name, *args, **kwargs):
        # [MIX 3] during a mixed step, the train logs are named per source (train_osse_mse, train_ose_mse, ...)
        if self._log_tag and name.startswith("train_"):
            name = f"train_{self._log_tag}_{name[len('train_'):]}"
        return super().log(name, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        # [MIX 4] one data source: unchanged. Two sources (dict from MixedOsseOseDataModule): one loss
        # per source, each averaged over its own pixels, then a weighted sum (so the dense OSSE pixels
        # do not swamp the sparse OSE ones)
        if not isinstance(batch, dict):
            return super().training_step(batch, batch_idx)
        weights = {"osse": self.osse_weight, "ose": 1.0 - self.osse_weight}
        grad_mode, total, wsum = self.grad_mode, 0.0, 0.0
        try:
            for name, sub in batch.items():
                self._log_tag = name
                self.grad_mode = self.osse_grad_mode if name == "osse" else grad_mode
                loss = self.step(sub, "train")[0]
                if loss is None:                       # batch skipped (too few target pixels)
                    self._n_rejected_batches = getattr(self, "_n_rejected_batches", 0) + 1
                    continue
                total, wsum = total + weights[name] * loss, wsum + weights[name]
        finally:
            self._log_tag, self.grad_mode = None, grad_mode
        if wsum == 0:
            return None
        loss = total / wsum
        self.log("train_loss_mixed", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def loss_mse(self, batch, out, phase):
        # [GRAD 3] "pairs": same value term, gradient term from neighbouring target pairs.
        # Also keeps the share of target cells that the gradient term uses (logged in step).
        w = self.get_rec_weight(phase)
        tgt_ok = batch.tgt.isfinite() & (w[None] > 0)
        if self.grad_mode == "pairs":
            loss = self.weighted_mse(out - batch.tgt, w)
            grad_loss, used = pair_grad_mse(out, batch.tgt, w)
        else:
            loss, grad_loss = super().loss_mse(batch, out, phase)
            with torch.no_grad():
                used = kfilts.sobel(batch.tgt).isfinite()
        self._grad_frac = (used & tgt_ok).sum() / tgt_ok.sum().clamp(min=1)
        return loss, grad_loss

    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < self.min_tgt_frac:
            return None, None
        # [HELDOUT 6] skip a batch that has no supervised pixel at all (in validation
        # this happens if the only held-out satellite has no data in the patch);
        # otherwise weighted_mse returns its 1000.0 placeholder and pollutes val_mse
        if not (batch.tgt.isfinite() & (self.get_rec_weight(phase) != 0)).any():
            return None, None
        _, out = self.base_step(batch, phase)
        loss_mse = self.loss_mse(batch, out, phase)
        training_loss = self.w_mse * loss_mse[0] + self.w_grad_mse * loss_mse[1]
        self.log(f"{phase}_gloss", loss_mse[1], prog_bar=False, on_step=False, on_epoch=True)
        # [GRAD 4] share of target cells used by the gradient term (both modes)
        self.log(f"{phase}_grad_frac", self._grad_frac.float(), prog_bar=False, on_step=False, on_epoch=True)
        return training_loss, out