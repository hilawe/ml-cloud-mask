"""
Build an augmented training feature table for the cloud-mask retrain.

Pulls PATMOS-x v06r00 daily granules from S3 (anonymous), caches each granule
locally to data/cache/, builds the 10-feature data frame for each target day
(day-before, day-of, day-after as in the production inference path), and writes
the concatenated, subsampled table as parquet.

Run:
    .venv/bin/python scripts/build_training_set.py \
        --days 2000-01-10,2000-07-15 \
        --satellite NOAA-14 --ascdes des \
        --sample-frac 0.10 \
        --out data/training_jan_jul_2000.parquet
"""

import argparse
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from skimage import filters


FEATURE_ORDER = ["t1", "t2", "t3", "tclr", "sobel", "snoice", "sfc", "t21", "t23", "dt"]
LAT_INDICES = (100, 1700)
LON_INDICES = (1, 3600)


def patmos_url(d: date, satellite: str, ascdes: str) -> str:
    yyyymmdd = d.strftime("%Y%m%d")
    s3 = s3fs.S3FileSystem(anon=True)
    glob = (
        f"noaa-cdr-patmosx-radiances-and-clouds-pds/data/{d.year}/"
        f"patmosx_v06r00_{satellite}_{ascdes}_d{yyyymmdd}_*nc"
    )
    hits = s3.glob(glob)
    assert hits, f"no PATMOS file found for {glob}"
    return "s3://" + hits[0]


def cached_download(url: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, os.path.basename(url))
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local
    print(f"  downloading {url} -> {local}")
    s3 = s3fs.S3FileSystem(anon=True)
    s3.get(url, local)
    return local


def open_local(path: str) -> xr.Dataset:
    y1, y2 = LAT_INDICES
    x1, x2 = LON_INDICES
    ds = xr.open_dataset(path)
    return ds.isel(longitude=slice(x1, x2), latitude=slice(y1, y2), time=0)


def features_for_day(d: date, satellite: str, ascdes: str, cache_dir: str) -> pd.DataFrame:
    d_b = d - timedelta(days=1)
    d_a = d + timedelta(days=1)
    url_b = patmos_url(d_b, satellite, ascdes)
    url_n = patmos_url(d, satellite, ascdes)
    url_a = patmos_url(d_a, satellite, ascdes)
    print(f"day {d}:")
    p_b = cached_download(url_b, cache_dir)
    p_n = cached_download(url_n, cache_dir)
    p_a = cached_download(url_a, cache_dir)

    ds1 = open_local(p_b)
    ds2 = open_local(p_n)
    ds3 = open_local(p_a)

    t1 = ds1["temp_11_0um_nom"].data
    t2 = ds2["temp_11_0um_nom"].data
    t3 = ds3["temp_11_0um_nom"].data
    tclr = ds2["temp_11_0um_nom_clear_sky"].data
    snoice = ds2["snow_class"].data
    sfc = ds2["land_class"].data
    cf = ds2["cloud_fraction"].data

    sobel = filters.sobel(t2)

    lat = ds2["latitude"].data
    lat2d = np.broadcast_to(lat[:, None], (lat.size, ds2["longitude"].size))

    df = pd.DataFrame({
        "t1": t1.flatten(),
        "t2": t2.flatten(),
        "t3": t3.flatten(),
        "tclr": tclr.flatten(),
        "sobel": sobel.flatten(),
        "snoice": snoice.flatten(),
        "sfc": sfc.flatten(),
        "t21": (t2 - t1).flatten(),
        "t23": (t2 - t3).flatten(),
        "dt": (tclr - t2).flatten(),
        "cf": cf.flatten(),
        "lat": lat2d.flatten(),
    })
    df["snoice"] = df["snoice"].replace({0: -1, 1: 0, 2: 1, 3: 1})
    df["sfc"] = df["sfc"].replace({3: 1, 4: 1, 5: 0, 6: 0, 7: 0})
    df["day"] = d.isoformat()

    n_before = len(df)
    df = df.dropna()
    print(f"  {n_before:,} pixels, {len(df):,} after dropna ({100 * len(df)/n_before:.1f}%)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="2000-01-10,2000-07-15",
                    help="comma-separated YYYY-MM-DD list")
    ap.add_argument("--satellite", default="NOAA-14")
    ap.add_argument("--ascdes", default="des")
    ap.add_argument("--sample-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out", default="data/training_jan_jul_2000.parquet")
    args = ap.parse_args()

    days = [date.fromisoformat(s) for s in args.days.split(",")]
    rng = np.random.default_rng(args.seed)

    frames = []
    for d in days:
        df = features_for_day(d, args.satellite, args.ascdes, args.cache_dir)
        # Subsample to keep training time reasonable while preserving cf distribution.
        n_keep = int(len(df) * args.sample_frac)
        idx = rng.choice(len(df), size=n_keep, replace=False)
        sub = df.iloc[idx].reset_index(drop=True)
        print(f"  subsampled to {len(sub):,} (frac={args.sample_frac})")
        frames.append(sub)

    all_df = pd.concat(frames, ignore_index=True)
    print(f"\ntotal training rows: {len(all_df):,}")
    print(f"cf distribution: mean={all_df['cf'].mean():.3f}, "
          f"frac(cf>0.5)={float((all_df['cf'] > 0.5).mean()):.3f}, "
          f"frac(cf>0.2)={float((all_df['cf'] > 0.2).mean()):.3f}")
    print(f"tropical land (lat in +/-15, sfc=1) fraction: "
          f"{float(((all_df['lat'].between(-15, 15)) & (all_df['sfc'] == 1)).mean()):.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_df.to_parquet(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
