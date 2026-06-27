"""
Tropical cirrus diagnostic on a single PATMOS-x granule.

Pulls one PATMOS-x v06r00 descending granule from the public Amazon Simple
Storage Service (S3) bucket (anonymous access, no credentials), builds the
ten model features over the tropical land strip (plus or minus 15 degrees
latitude, surface class equals 1), runs the saved XGBoost classifier in
probability mode, then stratifies the model's cloud probability by the
PATMOS-x cloud fraction (the truth used at training).

The diagnostic answers the question this project was started to answer.
If the model probability rises sharply once the PATMOS-x cloud fraction
crosses 0.5 but stays low for cf in (0.1, 0.5), the binary label is the
dominant lever and a lower training threshold (cf > 0.2 or similar) is the
cheapest intervention. If the model probability is flat across the (0,
0.5) cf band, the ten-feature window-channel set itself cannot separate
thin tropical cirrus from clear over warm land, and the route is a
feature-set change (for example adding ERA5 column water vapor).

Default day is 1999-07-15 NOAA-14 descending node, which is in the microwave land-surface emissivity
1998 surrounding-year window and over central Africa and the Western
Pacific in tropical convective season.
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from skimage import filters


FEATURE_ORDER = ["t1", "t2", "t3", "tclr", "sobel", "snoice", "sfc", "t21", "t23", "dt"]


def patmos_url(year, mmdd, satellite, ascdes):
    yyyymmdd = f"{year}{mmdd}"
    s3 = s3fs.S3FileSystem(anon=True)
    glob = (
        f"noaa-cdr-patmosx-radiances-and-clouds-pds/data/{year}/"
        f"patmosx_v06r00_{satellite}_{ascdes}_d{yyyymmdd}_*nc"
    )
    hits = s3.glob(glob)
    assert hits, f"no PATMOS file found for {glob}"
    return "s3://" + hits[0]


def open_patmos(url, lat_indices=(100, 1700), lon_indices=(1, 3600)):
    s3 = s3fs.S3FileSystem(anon=True)
    ds = xr.open_dataset(s3.open(url, "rb"))
    y1, y2 = lat_indices
    x1, x2 = lon_indices
    return ds.isel(longitude=slice(x1, x2), latitude=slice(y1, y2), time=0)


def build_features(ds1, ds2, ds3):
    t1 = ds1["temp_11_0um_nom"].data
    t2 = ds2["temp_11_0um_nom"].data
    t3 = ds3["temp_11_0um_nom"].data
    tclr = ds2["temp_11_0um_nom_clear_sky"].data
    snoice = ds2["snow_class"].data
    sfc = ds2["land_class"].data
    cf = ds2["cloud_fraction"].data

    sobel = filters.sobel(t2)

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
    })
    df["snoice"] = df["snoice"].replace({0: -1, 1: 0, 2: 1, 3: 1})
    df["sfc"] = df["sfc"].replace({3: 1, 4: 1, 5: 0, 6: 0, 7: 0})
    return df


def add_latitudes(ds2, df):
    lat = ds2["latitude"].data
    lat2d = np.broadcast_to(lat[:, None], (lat.size, ds2["longitude"].size))
    df["lat"] = lat2d.flatten()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=1999)
    ap.add_argument("--mmdd", default="0715")
    ap.add_argument("--satellite", default="NOAA-14")
    ap.add_argument("--ascdes", default="des")
    ap.add_argument("--model", default="model/xgboost_model_2.pkl")
    args = ap.parse_args()

    # Day-before and day-after URLs for the t1 / t3 features.
    from datetime import date, timedelta
    d0 = date(args.year, int(args.mmdd[:2]), int(args.mmdd[2:]))
    d_b = d0 - timedelta(days=1)
    d_a = d0 + timedelta(days=1)
    url_b = patmos_url(d_b.year, f"{d_b.month:02d}{d_b.day:02d}", args.satellite, args.ascdes)
    url_n = patmos_url(d0.year, args.mmdd, args.satellite, args.ascdes)
    url_a = patmos_url(d_a.year, f"{d_a.month:02d}{d_a.day:02d}", args.satellite, args.ascdes)
    print(f"day-before: {url_b}")
    print(f"day-of   : {url_n}")
    print(f"day-after: {url_a}")

    ds1 = open_patmos(url_b)
    ds2 = open_patmos(url_n)
    ds3 = open_patmos(url_a)

    df = build_features(ds1, ds2, ds3)
    df = add_latitudes(ds2, df)

    with open(args.model, "rb") as f:
        model = pickle.load(f)
    probs = model.predict_proba(df[FEATURE_ORDER])[:, 1]
    df["p_cloud"] = probs

    # Filter to tropical (plus or minus 15 degrees) land (sfc == 1), and drop NaN.
    tropical_land = df[
        (df["lat"].between(-15, 15)) & (df["sfc"] == 1) & df["t2"].notna() & df["tclr"].notna()
    ].copy()
    print(f"tropical land pixels: {len(tropical_land):,}")
    if len(tropical_land) == 0:
        return

    cf_bins = [0.0, 0.001, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01]
    cf_labels = ["cf=0", "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.7", "0.7-1"]
    tropical_land["cf_bin"] = pd.cut(tropical_land["cf"], bins=cf_bins, labels=cf_labels, right=False)

    by_bin = tropical_land.groupby("cf_bin", observed=False).agg(
        n=("p_cloud", "size"),
        p_mean=("p_cloud", "mean"),
        p_p25=("p_cloud", lambda s: s.quantile(0.25)),
        p_p50=("p_cloud", lambda s: s.quantile(0.50)),
        p_p75=("p_cloud", lambda s: s.quantile(0.75)),
        dt_mean=("dt", "mean"),
        frac_pred_cloudy_at_0p5=("p_cloud", lambda s: float((s > 0.5).mean())),
        frac_pred_cloudy_at_0p3=("p_cloud", lambda s: float((s > 0.3).mean())),
        frac_pred_cloudy_at_0p2=("p_cloud", lambda s: float((s > 0.2).mean())),
    )
    print()
    print("Tropical land, model p(cloud) stratified by PATMOS-x cloud_fraction:")
    print(by_bin.to_string())

    # Same diagnostic but ocean and snow-free, for the contrast.
    tropical_ocean = df[
        (df["lat"].between(-15, 15)) & (df["sfc"] == 0) & df["t2"].notna() & df["tclr"].notna()
    ].copy()
    tropical_ocean["cf_bin"] = pd.cut(tropical_ocean["cf"], bins=cf_bins, labels=cf_labels, right=False)
    by_bin_oc = tropical_ocean.groupby("cf_bin", observed=False).agg(
        n=("p_cloud", "size"),
        p_mean=("p_cloud", "mean"),
        frac_pred_cloudy_at_0p5=("p_cloud", lambda s: float((s > 0.5).mean())),
    )
    print()
    print("Tropical ocean (for contrast):")
    print(by_bin_oc.to_string())


if __name__ == "__main__":
    main()
