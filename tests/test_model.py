"""
Tests for the fraud model (step 3).

The sample data comes from the real step 1 generator and the real Spark
silver/gold code, so if someone changes the gold table in a way that breaks
the model, these tests catch it.
"""

import mlflow
import pandas as pd
import pytest

from model.train import (
    CHAMPION_ALIAS,
    MEDIANS_ARTIFACT,
    MODEL_PARAMS,
    REGISTERED_MODEL,
    build_model,
    fraud_scores,
    load_features,
    pick_threshold,
    split,
)

# features_path and trained_registry live in conftest.py, shared with the API tests.


@pytest.fixture(scope="module")
def data(features_path):
    return load_features(features_path)


@pytest.fixture(scope="module")
def trained(trained_registry):
    """Point MLflow at the test registry for this module. Returns the metrics per model."""
    mlflow.set_tracking_uri(trained_registry.tracking_uri)
    yield trained_registry.metrics
    mlflow.set_tracking_uri(None)


# --- Data --------------------------------------------------------------------

def test_load_features_fails_clearly_on_missing_column(tmp_path, data):
    data.drop(columns=["report_delay_days"]).to_parquet(tmp_path / "broken.parquet")
    with pytest.raises(ValueError, match="report_delay_days"):
        load_features(tmp_path / "broken.parquet")


def test_split_keeps_fraud_rate(data):
    _, _, y_train, y_test = split(data)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.01)


# --- Models ------------------------------------------------------------------

@pytest.mark.parametrize("name", list(MODEL_PARAMS))
def test_model_trains_and_scores_between_0_and_1(data, name):
    X_train, X_test, y_train, _ = split(data)
    scores = fraud_scores(build_model(name).fit(X_train, y_train), X_test)

    assert len(scores) == len(X_test)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_threshold_flags_about_the_review_rate(data):
    X_train, _, y_train, _ = split(data)
    model = build_model("logistic_regression").fit(X_train, y_train)
    threshold = pick_threshold(model, X_train, review_rate=0.1)

    flagged_rate = (fraud_scores(model, X_train) >= threshold).mean()
    assert flagged_rate == pytest.approx(0.1, abs=0.01)


# --- MLflow ------------------------------------------------------------------

def test_run_logs_every_model(trained):
    assert set(trained) == set(MODEL_PARAMS)
    for metrics in trained.values():
        assert 0 <= metrics["pr_auc"] <= 1
        assert "threshold" in metrics


def test_champion_is_registered_with_its_threshold(trained):
    version = mlflow.MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
    assert 0 < float(version.tags["threshold"]) < 1
    assert version.tags["model_type"] in MODEL_PARAMS


def test_registered_model_loads_and_scores_one_claim(trained):
    """What the API in step 4 will do: load the champion by name and score one raw claim."""
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@{CHAMPION_ALIAS}")

    # Every number is a decimal: the signature says double, and MLflow refuses
    # to convert a whole number (40) by itself. In step 4, FastAPI's request
    # model declares these fields as float, which turns 40 into 40.0 for us.
    claim = pd.DataFrame([{
        "product": "auto", "region": "Madrid", "channel": "web",
        "customer_age": 40.0, "claim_amount": 5200.0, "annual_premium": 550.0,
        "days_since_policy_start": 12.0, "report_delay_days": 25.0,
        "amount_vs_product_median": 3.1, "amount_vs_premium": 9.45,
    }])
    scores = model.predict(claim)           # predict_proba: [[P(honest), P(fraud)]]

    assert scores.shape == (1, 2)
    assert 0 <= scores[0][1] <= 1


def test_champion_has_medians_that_rebuild_the_gold_feature(trained, data):
    """
    The API rebuilds amount_vs_product_median from these medians. If they
    didn't reproduce the gold column exactly, the API would feed the model
    numbers it never saw in training.
    """
    version = mlflow.MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
    medians = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/{MEDIANS_ARTIFACT}")

    rebuilt = (data["claim_amount"] / data["product"].map(medians)).round(3)
    assert (rebuilt == data["amount_vs_product_median"]).all()
