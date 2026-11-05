"""Chronological splits. No random splitting anywhere in the project."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .panel import Panel


@dataclass(frozen=True)
class Fold:
    name: str
    val_start: int      # first validation week index
    test_start: int     # first test week index
    H: int              # forecast horizon (= test length)
    T: int

    @property
    def test_origin(self) -> int:
        return self.test_start - 1

    @property
    def test_targets(self) -> range:
        return range(self.test_start, self.test_start + self.H)

    def train_origins(self, stride: int, cutoff: int, min_origin: int = 8) -> list[int]:
        """Origins anchored at the latest admissible one, going back by ``stride``.
        Only rows whose target index is < cutoff are used for training (applied later via horizon filtering)."""
        last = cutoff - 2
        return sorted(range(last, min_origin - 1, -stride))

    def val_origins(self, step: int = 4) -> list[int]:
        out, t0 = [], self.val_start - 1
        while t0 <= self.test_start - 2:
            out.append(t0); t0 += step
        return out


def make_fold(panel: Panel, name: str, test_start_date: str, H: int, val_weeks: int) -> Fold:
    ts = panel.week_index(test_start_date)
    if ts + H > panel.T:
        raise ValueError(f"fold {name}: test window runs past the end of the data")
    if ts - val_weeks < 60:
        raise ValueError(f"fold {name}: too little history before validation window")
    return Fold(name=name, val_start=ts - val_weeks, test_start=ts, H=H, T=panel.T)
