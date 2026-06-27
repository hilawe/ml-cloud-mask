# Model summary

A short read of the model and the inference path.

## What the model is

A gradient-boosted binary classifier that reproduces the Pathfinder Atmospheres
Extended (PATMOS-x) cloud fraction from a thermal infrared (IR) window channel
plus light auxiliary surface and edge context. The intent is to take a
single-channel IR image (the 11 micron window) and emit a binary clear-or-cloudy
label per pixel that closely matches the PATMOS-x multi-channel cloud fraction.
The same model can then be applied to GridSat-B1, which carries the same 11
micron window channel but none of the other PATMOS-x inputs.

Hyperparameters of the production model:

- `XGBClassifier`, objective `binary:logistic`
- 900 boosting rounds, max depth 7, learning rate 0.01
- 10 input features
- held-out PATMOS-x accuracy about 90 percent at the default 0.5 threshold

## Inputs (features)

All ten features are scalar per pixel.

| feature  | definition |
|----------|------------|
| `t1`     | 11 micron brightness temperature, day before |
| `t2`     | 11 micron brightness temperature, current day |
| `t3`     | 11 micron brightness temperature, day after |
| `tclr`   | clear-sky 11 micron brightness temperature, current day |
| `sobel`  | Sobel edge response on the current-day 11 micron field |
| `snoice` | snow class collapsed to three bins, missing -1, no-snow 0, snow or sea ice 1 |
| `sfc`    | land class collapsed to three bins, 0 water, 1 land, 2 coast |
| `t21`    | `t2 - t1` |
| `t23`    | `t2 - t3` |
| `dt`     | `tclr - t2` |

Region for training and inference: the spatial domain is sliced to longitude
indices 1 to 3600 and latitude indices 100 to 1700 of the PATMOS-x 1800 by 3600
0.1 degree grid, which is roughly minus 80 to plus 80 degrees latitude. PATMOS-x
carries `temp_11_0um_nom` for ascending and descending nodes, and the inference
loop runs one pass at a time.

## Label

`cfbin = (cloud_fraction > 0.5).astype(int)`, where `cloud_fraction` is the
PATMOS-x value on the current day. Thin cirrus that PATMOS-x rates between 0 and
0.5 is therefore labelled clear during training, which is one of the levers for
the tropical cirrus question (see `Soft_Label_Experiment.md`).

## Feature importance, as observed in the saved model

Read from the trained model with `Booster.get_score`.

| feature  | gain   | weight (split count) |
|----------|--------|----------------------|
| `dt`     | 845.12 | 13094                |
| `tclr`   | 366.44 | 11902                |
| `snoice` | 167.42 | 2283                 |
| `sfc`    | 153.24 | 5130                 |
| `t2`     | 147.90 | 13949                |
| `sobel`  | 99.68  | 10886                |
| `t3`     | 30.74  | 9788                 |
| `t23`    | 29.17  | 10683                |
| `t21`    | 22.75  | 9672                 |
| `t1`     | 20.88  | 9244                 |

The story is sharp. The clear-sky departure `dt = tclr - t2` carries roughly 2.3
times the gain of the next feature, and 5 to 40 times the gain of the bare
temperatures and finite differences. The temporal context features (`t1`, `t3`,
`t21`, `t23`) are the weakest by a large margin. The surface context (`snoice`,
`sfc`) and the edge response are middling but well used. The mask is, in effect,
a learned cold-anomaly threshold operating on `tclr - t2`, with surface type,
snow or ice, and an edge prior softening the decision boundary.

## Training pipeline

1. Read three consecutive daily PATMOS-x granules from the public
   `noaa-cdr-patmosx-radiances-and-clouds-pds` Amazon Simple Storage Service (S3)
   bucket (anonymous access, no credentials required): the day before, the
   target day, and the day after.
2. Build the 10-feature data frame as above, including the binary label.
3. Drop rows with NaN values (mostly polar edges and missing scans).
4. `train_test_split` with a held-out fraction for evaluation.
5. Fit an `XGBClassifier` with the hyperparameters above.
6. Evaluate accuracy on the held-out fraction.

On a single global training day (about 1.1 million pixels) the training and
testing confusion matrices are essentially the same:

| outcome     | training | testing |
|-------------|----------|---------|
| clear (TN)  | 0.28     | 0.27    |
| cloud (TP)  | 0.63     | 0.63    |
| false clear | 0.047    | 0.047   |
| false cloud | 0.047    | 0.047   |

Overall accuracy is about 91 percent, with false-clear and false-cloud nearly
symmetric at the default 0.5 threshold. Against the legacy single-channel ISCCP
cloud mask (the algorithm GridSat-B1 inherits, Knapp 2011), with PATMOS-x as
truth, accuracy rises from about 71 to 92 percent, driven mostly by cutting the
legacy false-clear rate from about 28 percent to about 4.5 percent. So the model
is a substantial improvement on the legacy single-channel detector. The tropical
thin-cirrus question is the remaining 4.5 percent false-clear residual.

## Inference and outputs

`cloud_model_test_5.py` is the production inference path. Given a year, month,
day, satellite, and ascending or descending node, it pulls the three relevant S3
files, builds the same 10-feature data frame on the same lat-lon window, runs
`predict`, reshapes back to 2D, sets pixels with missing `temp_11_0um_nom` to -1,
and writes an Xarray dataset with two variables to Zarr:

- `cld_frac`: the original PATMOS-x `cloud_fraction`
- `cld_frac_model`: the ML binary prediction (0 clear, 1 cloudy, -1 missing)

## Notes and limits

- The training data set is not stored with the repository; it is rebuilt from the
  public S3 bucket on each run.
- The model summary numbers are single-sample, not cross-validated, and there is
  no per-region or per-season breakdown in the imported code.
- The production inference path takes the binary `predict`. The model also
  exposes `predict_proba`, so a downstream consumer can threshold per use case
  instead of at the default 0.5 boundary.
