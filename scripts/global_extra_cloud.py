"""
Global v2-vs-v3 breakdown, how much extra cloud the soft label makes and where.

Answers a reviewer question, if the soft-label retrain catches thin tropical
cirrus, what is the collateral cloud elsewhere. Runs both models on a full
global PATMOS-x day with the real ten features (no approximation), then
stratifies the cloud fraction and the false-cloud rate by latitude band and by
surface type, so the extra cloud is localized.

False cloud here is the fraction of pixels the model calls cloud where PATMOS-x
reports almost no cloud (cloud_fraction < 0.1), the cleanest "extra cloud over
genuinely clear sky" measure.

Run:
    .venv/bin/python scripts/global_extra_cloud.py --date 1999-07-15
"""

import argparse
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from skimage import filters
import pickle

FEATURE_ORDER = ["t1", "t2", "t3", "tclr", "sobel", "snoice", "sfc", "t21", "t23", "dt"]


def patmos_url(d, satellite, ascdes):
    s3 = s3fs.S3FileSystem(anon=True)
    g = (f"noaa-cdr-patmosx-radiances-and-clouds-pds/data/{d.year}/"
         f"patmosx_v06r00_{satellite}_{ascdes}_d{d.strftime('%Y%m%d')}_*nc")
    hits = s3.glob(g)
    assert hits, f"no file {g}"
    return "s3://" + hits[0]


def cached(url, cache_dir="data/cache"):
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, os.path.basename(url))
    if not (os.path.exists(local) and os.path.getsize(local) > 0):
        s3fs.S3FileSystem(anon=True).get(url, local)
    return local


def openp(path, y1=100, y2=1700, x1=1, x2=3600):
    return xr.open_dataset(path).isel(longitude=slice(x1, x2), latitude=slice(y1, y2), time=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="1999-07-15")
    ap.add_argument("--satellite", default="NOAA-14")
    ap.add_argument("--ascdes", default="des")
    args = ap.parse_args()
    d = date.fromisoformat(args.date)

    u_b = patmos_url(d - timedelta(days=1), args.satellite, args.ascdes)
    u_n = patmos_url(d, args.satellite, args.ascdes)
    u_a = patmos_url(d + timedelta(days=1), args.satellite, args.ascdes)
    ds1, ds2, ds3 = openp(cached(u_b)), openp(cached(u_n)), openp(cached(u_a))

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
        "t1": t1.flatten(), "t2": t2.flatten(), "t3": t3.flatten(),
        "tclr": tclr.flatten(), "sobel": sobel.flatten(),
        "snoice": snoice.flatten(), "sfc": sfc.flatten(),
        "t21": (t2 - t1).flatten(), "t23": (t2 - t3).flatten(),
        "dt": (tclr - t2).flatten(), "cf": cf.flatten(),
        "lat": lat2d.flatten(),
    })
    df["snoice_raw"] = df["snoice"]
    df["snoice"] = df["snoice"].replace({0: -1, 1: 0, 2: 1, 3: 1})
    df["sfc"] = df["sfc"].replace({3: 1, 4: 1, 5: 0, 6: 0, 7: 0})
    df = df.dropna(subset=["t2", "tclr"]).copy()

    m2 = pickle.load(open("model/xgboost_model_2.pkl", "rb"))
    m3 = pickle.load(open("model/xgboost_v3_cf_gt_0p2.pkl", "rb"))
    df["p2"] = m2.predict(df[FEATURE_ORDER])
    df["p3"] = m3.predict(df[FEATURE_ORDER])

    truth_cloud = (df["cf"] > 0.5).astype(int)
    clearish = df["cf"] < 0.1

    def block(title, mask):
        sub = df[mask]
        if len(sub) == 0:
            return
        tc = float((sub["cf"] > 0.5).mean())
        v2 = float((sub["p2"] == 1).mean())
        v3 = float((sub["p3"] == 1).mean())
        # False cloud over genuinely clear (cf < 0.1).
        cl = sub[sub["cf"] < 0.1]
        fc2 = float((cl["p2"] == 1).mean()) if len(cl) else float("nan")
        fc3 = float((cl["p3"] == 1).mean()) if len(cl) else float("nan")
        print(f"  {title:24s} n={len(sub):>9,}  PATMOS_cloud={tc:.3f}  "
              f"v2={v2:.3f} v3={v3:.3f} (d={v3-v2:+.3f})   "
              f"falsecloud@cf<0.1 v2={fc2:.3f} v3={fc3:.3f} (d={fc3-fc2:+.3f})")

    print(f"\nGlobal day {args.date} {args.satellite} {args.ascdes}, "
          f"valid pixels {len(df):,}\n")
    print("Cloud fraction and false-cloud-over-clear, v2 vs v3 (d = v3 - v2):\n")
    block("GLOBAL", df["t2"].notna())

    print("\n by latitude band:")
    block("tropics |lat|<23.5", df["lat"].abs() < 23.5)
    block("midlat 23.5-60", (df["lat"].abs() >= 23.5) & (df["lat"].abs() < 60))
    block("high-lat >60", df["lat"].abs() >= 60)

    print("\n by surface (sfc 0=water,1=land,2=coast):")
    block("ocean", df["sfc"] == 0)
    block("land", df["sfc"] == 1)
    block("coast", df["sfc"] == 2)

    print("\n tropics x surface (warm-ocean concern):")
    block("tropical ocean", (df["lat"].abs() < 23.5) & (df["sfc"] == 0))
    block("tropical land", (df["lat"].abs() < 23.5) & (df["sfc"] == 1))

    print("\n snow/ice present (snoice_raw in 2,3):")
    block("snow or ice surface", df["snoice_raw"].isin([2, 3]))

    # Global map of where v3 changes the mask versus v2.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid2d = np.isfinite(t2) & np.isfinite(tclr)
    idx = np.flatnonzero(valid2d.ravel())
    snoice_b = np.where(snoice == 0, -1, np.where(snoice == 1, 0, 1))
    sfc_b = np.select([np.isin(sfc, [3, 4]), np.isin(sfc, [5, 6, 7])],
                      [1, 0], default=sfc)
    feats = np.column_stack([
        t1.ravel()[idx], t2.ravel()[idx], t3.ravel()[idx], tclr.ravel()[idx],
        sobel.ravel()[idx], snoice_b.ravel()[idx], sfc_b.ravel()[idx],
        (t2 - t1).ravel()[idx], (t2 - t3).ravel()[idx], (tclr - t2).ravel()[idx],
    ])
    p2 = m2.predict(feats); p3 = m3.predict(feats)
    cat = np.full(t2.size, np.nan)
    flat = cat
    base = np.full(idx.size, np.nan)
    base[(p2 == 0) & (p3 == 0)] = 0   # agree clear
    base[(p2 == 1) & (p3 == 1)] = 3   # agree cloud
    base[(p2 == 0) & (p3 == 1)] = 2   # v3 adds cloud
    base[(p2 == 1) & (p3 == 0)] = 1   # v3 removes cloud
    flat[idx] = base
    cat2d = flat.reshape(t2.shape)

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = matplotlib.colors.ListedColormap(["#1f4ed8", "#22c1c1", "#f2c200", "#d62728"])
    im = ax.imshow(np.flipud(cat2d), aspect="auto", cmap=cmap, vmin=0, vmax=3,
                   extent=[-180, 180, float(lat.min()), float(lat.max())])
    ax.set_title(f"Where the soft label (v3) changes the mask vs v2, global {args.date}")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    cb = fig.colorbar(im, ax=ax, ticks=[0.375, 1.125, 1.875, 2.625], shrink=0.8)
    cb.ax.set_yticklabels(["agree clear", "v3 removes cloud", "v3 ADDS cloud", "agree cloud"])
    fig.tight_layout()
    os.makedirs("docs/figures", exist_ok=True)
    png = f"docs/figures/global_v3_vs_v2_{args.date}.png"
    fig.savefig(png, dpi=120)
    print(f"\n  saved global map: {png}")


if __name__ == "__main__":
    main()
