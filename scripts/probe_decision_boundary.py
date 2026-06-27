"""
Probe the saved XGBoost cloud-mask classifier with synthetic feature vectors
that mimic clear, opaque-cloud, and thin-tropical-cirrus regimes.

The point is to read the model's probability calibration without any
network access. If the model already assigns sensible cloud probabilities
to thin-cirrus-like inputs, then the lever is the binary threshold at
inference, post-hoc thresholding solves it without a retrain. If the
probabilities are also low for thin cirrus, then either the training label
(cf > 0.5 binarization) is throwing the thin-cirrus signal away, or the
feature set (a single 11 micron window plus context) cannot separate thin
cirrus from clear over a warm tropical surface in the first place.

Inputs the model expects, in order:
    t1, t2, t3, tclr, sobel, snoice, sfc, t21, t23, dt
"""

import pickle
import numpy as np
import pandas as pd

FEATURE_ORDER = ["t1", "t2", "t3", "tclr", "sobel", "snoice", "sfc", "t21", "t23", "dt"]


def make_row(*, t1, t2, t3, tclr, sobel, snoice, sfc):
    t21 = t2 - t1
    t23 = t2 - t3
    dt = tclr - t2
    return {
        "t1": t1, "t2": t2, "t3": t3, "tclr": tclr,
        "sobel": sobel, "snoice": snoice, "sfc": sfc,
        "t21": t21, "t23": t23, "dt": dt,
    }


# Build a small synthetic table that walks the decision space.
# All temperatures in kelvin; tclr is a plausible clear-sky 11 micron BT.
rows = []

# Clear tropical ocean, surface ~ tclr ~ 297 K, observed BT matches.
rows.append(("clear tropical ocean",        make_row(t1=296, t2=297, t3=297, tclr=297, sobel=0.4, snoice=0, sfc=0)))
# Clear tropical land, day-to-day variability.
rows.append(("clear tropical land",         make_row(t1=300, t2=302, t3=301, tclr=302, sobel=1.0, snoice=0, sfc=1)))
# Opaque deep-convective cloud, very cold top, large clear-sky departure.
rows.append(("opaque deep convection",      make_row(t1=298, t2=215, t3=298, tclr=298, sobel=4.0, snoice=0, sfc=0)))
# Mid-level water cloud over warm ocean.
rows.append(("mid-level water cloud",       make_row(t1=297, t2=275, t3=297, tclr=297, sobel=2.0, snoice=0, sfc=0)))
# Thin tropical cirrus over warm ocean, dt ~ 10 K (semi-transparent, warm cirrus).
rows.append(("thin cirrus over ocean (dt=10)",  make_row(t1=297, t2=287, t3=297, tclr=297, sobel=0.6, snoice=0, sfc=0)))
# Thinner cirrus, dt ~ 5 K (near the noise level a window-channel mask can see).
rows.append(("very thin cirrus ocean (dt=5)",   make_row(t1=297, t2=292, t3=297, tclr=297, sobel=0.5, snoice=0, sfc=0)))
# Thin tropical cirrus over warm land.
rows.append(("thin cirrus over land (dt=10)",   make_row(t1=302, t2=292, t3=302, tclr=302, sobel=0.6, snoice=0, sfc=1)))
# Snow surface, cold but clear.
rows.append(("clear snow surface (winter)", make_row(t1=255, t2=255, t3=255, tclr=255, sobel=0.3, snoice=1, sfc=1)))
# Cloud over snow, similar BT but cold-anomaly small.
rows.append(("cloud over snow surface",     make_row(t1=255, t2=240, t3=255, tclr=255, sobel=1.0, snoice=1, sfc=1)))
# Coastline, mixed signal.
rows.append(("clear coastline",             make_row(t1=295, t2=297, t3=296, tclr=297, sobel=2.0, snoice=0, sfc=2)))


def main():
    with open("model/xgboost_model_2.pkl", "rb") as f:
        model = pickle.load(f)

    labels = [name for name, _ in rows]
    df = pd.DataFrame([r for _, r in rows])[FEATURE_ORDER]

    probs = model.predict_proba(df)[:, 1]
    preds_default = (probs > 0.5).astype(int)

    print(f"{'case':38s}  {'dt':>5s}  {'t2':>5s}  {'tclr':>5s}  "
          f"{'p(cloud)':>9s}  {'cls@.5':>6s}  {'cls@.3':>6s}  {'cls@.2':>6s}  {'cls@.1':>6s}")
    print("-" * 110)
    for name, row, p, cls in zip(labels, df.to_dict("records"), probs, preds_default):
        print(f"{name:38s}  {row['dt']:5.1f}  {row['t2']:5.1f}  {row['tclr']:5.1f}  "
              f"{p:9.3f}  {cls:>6d}  "
              f"{int(p > 0.3):>6d}  {int(p > 0.2):>6d}  {int(p > 0.1):>6d}")


if __name__ == "__main__":
    main()
