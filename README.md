# Machine-learning cloud mask from a single infrared window channel

A gradient-boosted classifier that reproduces a modern multi-channel cloud
product from a single thermal-infrared window channel. It is trained on the
Pathfinder Atmospheres Extended (PATMOS-x) cloud fraction as truth and learns to
emit a clear-or-cloudy label per pixel from the 11 micron brightness temperature
plus light surface and edge context. Because it needs only the one window
channel at inference, the same model can be applied to the long single-channel
satellite record (ISCCP and GridSat-B1, which reach back to the early 1980s) so
that record gains the skill of the modern multi-channel mask.

The framing is "can the old record learn from the new." Train on the modern
multi-channel Climate Data Record (CDR), then apply to the legacy single-channel
imagery so the long record inherits the better mask.

## Authors and provenance

The model was co-developed by Hilawe Semunegus (NOAA National Centers for
Environmental Information, NCEI), Ken Knapp (Knapp WeatherSat Services LLC,
formerly NOAA NCEI), and Xuepeng (Tom) Zhao (North Carolina Institute for
Climate Studies, NCICS) in the NCICS Earth Science Data Science course. The
GridSat-B1 record it screens is a NOAA Climate Data Record (Knapp 2011).

## How it works

The classifier is an `XGBClassifier` (objective `binary:logistic`, 900 boosting
rounds, max depth 7, learning rate 0.01) over ten scalar per-pixel features:

| feature | meaning |
|---------|---------|
| `t1`, `t2`, `t3` | 11 micron brightness temperature for the day before, the target day, and the day after |
| `tclr` | clear-sky 11 micron brightness temperature on the target day |
| `sobel` | Sobel edge response on the target-day window field |
| `snoice` | snow or sea-ice class, collapsed to three bins |
| `sfc` | surface class, water / land / coast |
| `t21`, `t23` | `t2 - t1` and `t2 - t3` |
| `dt` | `tclr - t2`, the clear-sky departure |

By feature importance the model is, in effect, a learned cold-anomaly threshold
on the clear-sky departure `dt`. That one feature carries about 2.3 times the
gain of the next, with surface type, snow or ice, and the edge response
softening the boundary, and the day-to-day temporal context contributing little.
The training label is the PATMOS-x cloud fraction binarized at 0.5.

## Performance

On a held-out PATMOS-x sample the model runs about 91 percent accurate at the
default 0.5 threshold, with false-clear and false-cloud rates near 5 percent each
and roughly symmetric. Measured against the legacy single-channel ISCCP cloud
mask with PATMOS-x as truth, it raises accuracy from about 71 to 92 percent,
driven mostly by cutting the legacy false-clear rate from about 28 percent to
about 4.5 percent. The remaining false-clear residual is concentrated in thin
cirrus over the warm, humid tropics, which motivated the soft-label experiment
below.

## The thin-cirrus soft-label experiment

The mask is used downstream as the clear-sky screen for a microwave land-surface
emissivity retrieval. That retrieval, validated against the Tool to Estimate
Land Surface Emissivities at Microwaves (TELSEM) climatology, left a residual at
85 GHz that tracked column water vapor, consistent with thin cirrus near deep
convection being passed as clear. Since the training label binarizes cloud
fraction at 0.5, thin cirrus that the truth product rates between 0.1 and 0.5 is
taught as clear, so the natural intervention is a softer label.

Retraining on the same ten features with the label softened to `cf > 0.2`
improves thin-cirrus detection in the 0.1 to 0.5 cloud-fraction belt by roughly
30 to 50 percent on PATMOS-x, at no cost in clear-land false positives, though it
does over-flag warm ocean and so is not a drop-in global replacement. The honest
result is that this gain does not carry to the single-channel domain: applied
through the window-channel features available on GridSat-B1, the soft label
re-screens under one percent of tropical-land pixels, because the cirrus driving
the residual sits only 1 to 3 kelvin below clear sky, near the noise floor of any
window-channel mask. The soft label stands as a general improvement to the mask
against PATMOS-x truth, not as the lever for the microwave residual, for which a
direct microwave scattering screen is the more promising path. `docs/` carries
the model summary and the full experiment write-up with the numbers.

## Layout

```
cloud_model_test_5.py    inference: PATMOS-x S3 inputs, build features, XGBoost predict, Zarr output
functions.py             S3 URL builders for PATMOS-x daily granules
scripts/                 training-table build, model fitting, decision-boundary and cirrus diagnostics
model/                   the trained classifiers (portable JSON and pickle)
docs/                    model summary and the soft-label experiment
tests/                   a smoke test that loads a model and predicts
```

## Usage

The models load with XGBoost from the portable JSON:

```python
import numpy as np, xgboost as xgb
m = xgb.XGBClassifier()
m.load_model("model/xgboost_v3_cf_gt_0p2.json")
# features in order: t1, t2, t3, tclr, sobel, snoice, sfc, t21, t23, dt
row = np.array([[280, 278, 279, 290, 0.1, 0, 1, 2, -1, 12]], dtype=float)
print(int(m.predict(row)[0]), float(m.predict_proba(row)[0, 1]))
```

`cloud_model_test_5.py` is the production inference path: given a date, satellite,
and node, it pulls three consecutive daily PATMOS-x granules, builds the
ten-feature table on the same window, predicts, and writes the result. The
PATMOS-x inputs are read from the public `noaa-cdr-patmosx-radiances-and-clouds-pds`
Amazon S3 bucket with anonymous access, so no credentials are needed.

## License

Released to the public domain under Creative Commons Zero 1.0 (see `LICENSE`).
