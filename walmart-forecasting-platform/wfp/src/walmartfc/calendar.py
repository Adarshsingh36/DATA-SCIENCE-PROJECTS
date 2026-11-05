"""Calendar features that are known in advance for any target week (no leakage)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.easter import easter

# Kaggle's IsHoliday weeks are the week-ending Fridays containing: Super Bowl (Feb), Labor Day (Sep),
# Thanksgiving (Nov), Christmas (Dec). The type is recovered from the flagged date's month.
HOL_TYPE = {2: 1, 9: 2, 11: 3, 12: 4}


def _next_date(d: pd.Timestamp, month: int, day: int) -> pd.Timestamp:
    c = pd.Timestamp(d.year, month, day)
    return c if c >= d - pd.Timedelta(days=3) else pd.Timestamp(d.year + 1, month, day)


def _thanksgiving(year: int) -> pd.Timestamp:
    nov1 = pd.Timestamp(year, 11, 1)
    first_thu = nov1 + pd.Timedelta(days=(3 - nov1.dayofweek) % 7)
    return first_thu + pd.Timedelta(weeks=3)


def calendar_frame(dates: pd.DatetimeIndex, is_holiday: np.ndarray) -> pd.DataFrame:
    """One row per week. ``is_holiday`` is Kaggle's flag, itself a calendar fact known in advance."""
    d = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=d)
    out["woy"] = d.isocalendar().week.to_numpy().astype(int)
    out["month"] = d.month.to_numpy()
    out["is_holiday"] = np.asarray(is_holiday).astype(int)
    out["hol_type"] = [HOL_TYPE.get(m, 0) if h else 0 for m, h in zip(d.month, out["is_holiday"])]
    out["wk_to_xmas"] = [((_next_date(x, 12, 25) - x).days) / 7.0 for x in d]
    thx = []
    for x in d:
        t = _thanksgiving(x.year)
        if t < x - pd.Timedelta(days=3):
            t = _thanksgiving(x.year + 1)
        thx.append((t - x).days / 7.0)
    out["wk_to_thx"] = thx
    ed = []
    for x in d:
        e = pd.Timestamp(easter(x.year))
        ed.append(np.clip((e - x).days / 7.0, -3, 3) if abs((e - x).days) <= 21 else np.nan)
    out["easter_dist"] = ed
    out["tgt_dom"] = d.day.to_numpy()
    return out
