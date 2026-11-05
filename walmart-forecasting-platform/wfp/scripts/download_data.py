"""Download the Walmart Store Sales Forecasting files into data/raw/.

Primary source (official, needs a free Kaggle account + accepted competition rules):
    kaggle competitions download -c walmart-recruiting-store-sales-forecasting
Kaggle serves train/features/stores as zipped CSVs inside the competition bundle.

Fallback: a public GitHub copy of the same three files (used when the Kaggle CLI is not configured).
The script verifies row counts so a wrong/partial file is never silently accepted.
"""
from __future__ import annotations

import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED = {"train.csv": 421570, "stores.csv": 45, "features.csv": 8190}
MIRROR = "https://raw.githubusercontent.com/ezgigm/Project4_Store_Sales_Forecasting/master/Raw%20Data/{}"


def ok() -> bool:
    for f, n in EXPECTED.items():
        p = RAW / f
        if not p.exists():
            return False
        rows = sum(1 for _ in open(p, "rb", buffering=1 << 20)) - 1 if f != "stores.csv" else len(p.read_bytes().split()) - 1
        if rows != n:
            print(f"{f}: expected {n} rows, found {rows}")
            return False
    return True


def via_kaggle() -> bool:
    try:
        RAW.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "competitions", "download", "-c", "walmart-recruiting-store-sales-forecasting", "-p", str(RAW)],
                       check=True)
        for z in RAW.glob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(RAW)
            z.unlink()
        for f in ("train.csv", "features.csv"):           # Kaggle ships these as nested zips
            zp = RAW / f"{f}.zip"
            if zp.exists():
                with zipfile.ZipFile(zp) as zf:
                    zf.extractall(RAW)
                zp.unlink()
        return ok()
    except Exception as e:  # noqa: BLE001
        print("Kaggle download unavailable:", e)
        return False


def via_mirror() -> bool:
    RAW.mkdir(parents=True, exist_ok=True)
    for f in EXPECTED:
        print("fetching", f)
        urllib.request.urlretrieve(MIRROR.format(f), RAW / f)
    return ok()


if __name__ == "__main__":
    if ok():
        print("data/raw already complete")
    elif via_kaggle() or via_mirror():
        print("data/raw ready")
    else:
        sys.exit("Could not obtain the dataset. Download it manually from "
                 "https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data into data/raw/")
