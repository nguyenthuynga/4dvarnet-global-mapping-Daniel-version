"""
Joint OSSE + OSE training: the same UNet as usual (15 days in, 15 days out), trained on
a 50/50 mix of two datasets. Every batch alternates OSSE and OSE patches, so a batch of
16 holds 8 OSSE + 8 OSE patches.

  OSSE patch : GLORYS. Input = GLORYS on a random-k union of the 2024 footprints,
               target = GLORYS on the full union ("gappy", like the real target).
  OSE patch  : real data. Input = a random-k union of the satellites,
               target = the union of all satellites.
Both are normalised with the same stats (datamodule.norm_stats, the GLORYS ones).
Validation: OSSE only (GLORYS over domains.val).


Reuses, unchanged: RealRandomOSSEDataset (real_random_osse.py) and RealRandomDataset
(real_random.py). Put this file in contrib/glorys12/.
"""
import numpy as np
import torch

from ocean4dvarnet.data import BaseDataModule, TrainingItem

from .real_random import RealRandomDataset
from .real_random_osse import RealRandomOSSEDataset
from .unet import NormParamsNotProvided


class _Seeded:
    """Random draws (satellite subset, noise) that differ between DataLoader workers and
    epochs; with fixed_seed (validation) the same draw for an item at every validation."""
    fixed_seed = None

    def __getitem__(self, item):
        if self.fixed_seed is not None:
            self._rng = np.random.default_rng([int(self.fixed_seed), int(item)])
        else:
            seed = torch.initial_seed()        # new value per worker and per epoch
            if getattr(self, "_rng_seed", None) != seed:
                self._rng_seed = seed
                self._rng = np.random.default_rng(seed % 2**32)
        return super().__getitem__(item)


class _OSSEDataset(_Seeded, RealRandomOSSEDataset):
    pass


class _OSEDataset(_Seeded, RealRandomDataset):
    pass


class _InputTarget(torch.utils.data.Dataset):
    """Same item type for both datasets: (input, tgt). lon/lat are not used by the UNet."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        it = self.ds[i]
        return TrainingItem(it.input, it.tgt)



def _normalizer(m, s):
    def post(item):
        return item._replace(input=(item.input - m) / s, tgt=(item.tgt - m) / s)
    return post


class _AlternatingSampler(torch.utils.data.Sampler):
    """OSSE, OSE, OSSE, OSE, ... : each patch drawn at random inside its own dataset.
    With an even batch size every batch is exactly half OSSE, half OSE. Indices are made
    on the fly, so limit_train_batches decides how many are used per epoch."""

    def __init__(self, n_osse, n_ose, num_samples, chunk=65536):
        self.n_osse, self.n_ose = int(n_osse), int(n_ose)
        self.num_samples, self.chunk = int(num_samples), int(chunk)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        left = self.num_samples
        while left > 0:
            n = min(self.chunk, left)
            half = (n + 1) // 2
            i_osse = torch.randint(self.n_osse, (half,))
            i_ose = self.n_osse + torch.randint(self.n_ose, (half,))
            yield from torch.stack([i_osse, i_ose], dim=1).flatten()[:n].tolist()
            left -= n


class JointOSSEOSEDataModule(BaseDataModule):
    """input_da = load_glorys12_osse_masks(...) (OSSE), ose_da = load_real_satellites(...)."""

    def __init__(self, *args, ose_da=None, ose_periods=None, mode_target="gappy",
                 k_min=2, k_max=7, ose_noise=None, **kwargs):
        self.ose_da = ose_da
        self.ose_periods = list(ose_periods or [])
        self.mode_target = mode_target
        self.k_min, self.k_max = k_min, k_max
        self.ose_noise = ose_noise
        super().__init__(*args, **kwargs)
        self.tgt, self.masks, self.target_idx = self.input_da
        assert self.ose_da is not None and self.ose_periods, "ose_da and ose_periods are needed"

    def norm_stats(self):
        if self._norm_stats is None:
            raise NormParamsNotProvided()
        return self._norm_stats

    def setup(self, stage="test"):
        (m, s), (mv, sv) = self.norm_stats()["train"], self.norm_stats()["val"]
        k = dict(k_min=self.k_min, k_max=self.k_max)

        osse_train = _OSSEDataset(
            self.tgt.sel(self.domains["train"]), self.masks, self.target_idx,
            **self.xrds_kw["train"], postpro_fn=_normalizer(m, s),
            mode_target=self.mode_target, **k)
        ose_kw = dict(self.xrds_kw["train"])
        if self.ose_noise is not None:
            ose_kw["noise"] = self.ose_noise
        ose_train = [_OSEDataset(self.ose_da.sel(p), **ose_kw, postpro_fn=_normalizer(m, s), **k)
                     for p in self.ose_periods]      # one dataset per continuous period
        self.train_ds = torch.utils.data.ConcatDataset(
            [_InputTarget(osse_train)] + [_InputTarget(d) for d in ose_train])
        self._n_osse = len(osse_train)
        self._n_ose = sum(len(d) for d in ose_train)

        osse_val = _OSSEDataset(
            self.tgt.sel(self.domains["val"]), self.masks, self.target_idx,
            **self.xrds_kw["val"], postpro_fn=_normalizer(mv, sv),
            mode_target=self.mode_target, **k)
        osse_val.fixed_seed = 0
        self.val_ds = _InputTarget(osse_val)
        print(f"[joint] train patches available: OSSE {self._n_osse:,}, OSE {self._n_ose:,} "
              f"({len(ose_train)} period(s)), drawn 50/50; val (OSSE) patches: {len(self.val_ds):,}")

    def train_dataloader(self):
        sampler = _AlternatingSampler(self._n_osse, self._n_ose, len(self.train_ds))
        return torch.utils.data.DataLoader(self.train_ds, sampler=sampler, **self.dl_kw)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, batch_size=1,
                                           num_workers=1)
