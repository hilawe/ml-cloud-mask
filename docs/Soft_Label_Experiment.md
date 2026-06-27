# Soft-label retrain for tropical thin cirrus

The mask is used downstream as the clear-sky screen for a microwave land-surface
emissivity retrieval. Validated against the Tool to Estimate Land Surface
Emissivities at Microwaves (TELSEM) climatology over a full annual cycle of
SSM/I data, that retrieval left a one-sided residual at 85 GHz that grew with
column water vapor. The 22 GHz water-vapor channel was unaffected, and a
viewing-angle test came back negative once the latitude confound was removed,
which points the residual at scattering by thin cirrus near deep convection that
the cloud mask passes as clear over the warm, humid tropics.

That diagnosis lands on a property of the training label. The mask binarizes
PATMOS-x cloud fraction at 0.5, so thin cirrus that the truth product rates
between 0.1 and 0.5 is taught as clear. The cheapest intervention is to soften
the label, and that is the experiment here.

## Setup

`scripts/build_training_set.py` builds a training table from PATMOS-x NOAA-14
descending granules (day before, day of, day after) for two target days, a
winter day and a summer day, so the model sees some tropical-summer cloud:

- 2000-01-10: about 5.3 million valid pixels, subsampled at 10 percent
- 2000-07-15: about 5.8 million valid pixels, subsampled at 10 percent

The concatenated table is 1,107,284 rows. `scripts/train_models.py` then fits two
classifiers on this same table with the production hyperparameters and the same
split, so any difference is attributable to the label, not the data:

- a baseline at the original `cf > 0.5`
- the soft label at `cf > 0.2`

## Held-out test

| outcome           | baseline cf>0.5 | soft-label cf>0.2 |
|-------------------|-----------------|-------------------|
| clear (TN)        | 0.2342          | 0.1500            |
| false cloud (FP)  | 0.0475          | 0.0501            |
| false clear (FN)  | 0.0467          | 0.0374            |
| cloud (TP)        | 0.6717          | 0.7626            |
| accuracy          | 0.9059          | 0.9125            |

The soft-label model trades a small rise in false cloud (4.8 to 5.0 percent) for
a larger drop in false clear (4.7 to 3.7 percent) and slightly higher overall
accuracy.

## Tropical-land cirrus diagnostic

`scripts/tropical_cirrus_diagnostic.py` on an independent day (1999-07-15
NOAA-14 descending, 230,236 tropical-land pixels). Each cell is the fraction of
pixels in that PATMOS-x cloud-fraction bin flagged cloudy at the 0.5 threshold.

| PATMOS-x cf | n       | v2 (saved) | baseline cf>0.5 | soft-label cf>0.2 |
|-------------|---------|------------|-----------------|-------------------|
| cf=0        | 65,581  | 0.034      | 0.006           | 0.037             |
| 0.1 to 0.2  | 7,791   | 0.166      | 0.050           | 0.237             |
| 0.2 to 0.3  | 5,976   | 0.257      | 0.097           | 0.354             |
| 0.3 to 0.5  | 11,229  | 0.402      | 0.214           | 0.523             |
| 0.5 to 0.7  | 11,170  | 0.623      | 0.452           | 0.726             |
| 0.7 to 1.0  | 128,489 | 0.960      | 0.937           | 0.972             |

Three readouts.

First, the soft label improves thin-cirrus detection where it was meant to. In
the 0.1 to 0.5 cloud-fraction belt the fraction flagged cloudy rises by roughly
30 to 50 percent relative to the saved model, and the mean predicted probability
for the 0.3 to 0.5 bin crosses 0.5. The clear-land false-positive rate (cf=0)
stays at 3.7 percent, basically unchanged.

Second, the data augmentation alone does not do it. The baseline at `cf > 0.5` on
the augmented table catches fewer marginal-belt pixels than the saved model, so
adding the summer day made the model more decisive at separating opaque cloud
from clear and less attentive to the marginal belt. The label is the
load-bearing change, not the data.

Third, the cost is concentrated over warm ocean. The cf=0 false-positive rate
over tropical ocean rises from about 2.5 to 23 percent. Over land it is bounded,
but a globally applied soft-label mask over-flags ocean, so the soft label is not
a drop-in global replacement. Two mitigations need no further retraining: use the
continuous `predict_proba` and raise the threshold over ocean, or route the soft
label only to the land consumer.

## Single-channel reality check

`scripts/gridsat_preview.py` sizes the effect on the single-channel domain the
downstream retrieval actually runs on, rather than on PATMOS-x where the model
was trained. On that domain only the dominant window-channel features are
available exactly (`dt`, `tclr`, `t2`, `sobel`), and the remaining four are
approximated, so the model-to-model comparison on identical features is clean
while an absolute comparison is read loosely.

Over 4.0 million tropical-land pixel-timesteps on a sample day, switching the
baseline to the soft label re-screens about 2 percent of the currently-clear
tropical-land pixels and un-screens a similar amount elsewhere, so the net cloud
fraction barely moves. The 30 to 50 percent relative PATMOS-x gain does not carry
over, for three reasons:

1. The PATMOS-x gain lives inside the 0.1 to 0.5 belt, a thin slice of all
   clear-sky pixels, so a large relative gain there is a small absolute change.
2. The cirrus driving the residual has a clear-sky departure of only 1 to 3
   kelvin, near the noise floor of a window-channel mask, so those pixels sit
   deep in the clear region and the softened boundary still misses most of them.
3. A full-feature single-channel application already flags more of that marginal
   cirrus than the approximate reconstruction, though this point is confounded by
   the approximation and held loosely.

## Conclusion

The soft-label retrain is a legitimate general improvement to the mask against
PATMOS-x truth, but it is not the lever for the microwave 85 GHz residual once
applied through a window-channel mask. The more promising and cheaper path for
that residual is a direct microwave scattering screen, for example an 85V minus
85H depression, which acts on the channels carrying the bias and does not depend
on the infrared cloud mask at all. Two follow-ups remain queued: per-surface-class
thresholding at inference, and adding total column water vapor as an eleventh
feature.
