"""Smoke test: the trained models load and predict on the ten-feature input.

Verifies the published model artifacts are loadable with XGBoost and behave as a
ten-feature binary classifier (clear vs cloud), so the public CI fails loudly if
a model file or the feature contract is broken.
"""

import os

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [
    "model/xgboost_model_2.json",
    "model/xgboost_v3_cf_gt_0p2.json",
]
# Feature order: t1, t2, t3, tclr, sobel, snoice, sfc, t21, t23, dt
N_FEATURES = 10
SAMPLE = np.array([
    [280.0, 278.0, 279.0, 290.0, 0.10, 0, 1, 2.0, -1.0, 12.0],  # warm clear-ish
    [235.0, 233.0, 234.0, 285.0, 0.40, 0, 1, 2.0, -1.0, 52.0],  # cold, large dt -> cloud-like
], dtype=float)


@pytest.mark.parametrize("rel", MODELS)
def test_model_loads_and_predicts(rel):
    path = os.path.join(HERE, rel)
    if not os.path.exists(path):
        pytest.skip(f"model not present: {rel}")
    m = xgb.XGBClassifier()
    m.load_model(path)
    assert int(getattr(m, "n_features_in_", N_FEATURES)) == N_FEATURES

    pred = m.predict(SAMPLE)
    assert pred.shape == (SAMPLE.shape[0],)
    assert set(np.unique(pred)).issubset({0, 1})

    proba = m.predict_proba(SAMPLE)[:, 1]
    assert proba.shape == (SAMPLE.shape[0],)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
