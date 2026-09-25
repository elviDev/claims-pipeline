"""
Tests for the fraud model (step 3).

The sample data comes from the real step 1 generator and the real Spark
silver/gold code, so if someone changes the gold table in a way that breaks
the model, these tests catch it.
"""

import os

import mlflow
import numpy as np
import pandas as pd
import pytest

from generate_claims import generate_claims, generate_policies
from model.train import (
    CHAMPION_ALIAS,
    MODEL_PARAMS,
    REGISTERED_MODEL,
    build_model,
    fraud_scores,
    load_features,
    pick_threshold,
    run,
    split,
)
from pipeline.bronze import read_raw_csv
from pipeline.gold import build_fraud_features
from pipeline.silver import build_silver_claims, build_silver_policies


@pytest.fixture(scope="module")
def features_path(spark, tmp_path_factory):
    """Generate a small dataset and run it through silver and gold. Returns the gold folder."""
    tmp = tmp_path_factory.mktemp("model_data")
    rng = np.random.default_rng(7)
    policies = generate_policies(300, rng)
    # No add_dirty_data here: the pipeline tests already cover that. The model only sees clean gold rows.
    generate_claims(policies, 2000, rng).to_csv(tmp / "claims.csv", index=False)
    policies.to_csv(tmp / "policies.csv", index=False)

    silver_policies, _, _ = build_silver_policies(read_raw_csv(spark, str(tmp / "policies.csv")))
    silver_claims, _, _ = build_silver_claims(read_raw_csv(spark, str(tmp / "claims.csv")), silver_policies)

    path = tmp / "fraud_features"
    build_fraud_features(silver_claims, silver_policies).write.parquet(str(path))
    return path


@pytest.fixture(scope="module")
def data(features_path):
    return load_features(features_path)


@pytest.fixture(scope="module")
def trained(features_path, tmp_path_factory):
    """
    Run the full training script once, in a temp folder.

    MLflow writes mlflow.db and mlruns/ into the current folder, so we move
    there first. Otherwise every test run would add fake runs to the real registry.
    """
    workdir = tmp_path_factory.mktemp("mlflow")
    old_cwd = os.getcwd()
    os.chdir(workdir)
    try:
        metrics = run(features_path, review_rate=0.1, tracking_uri="sqlite:///mlflow.db")
        yield metrics
    finally:
        os.chdir(old_cwd)
        mlflow.set_tracking_uri(None)  # back to the default, so later tests aren't affected


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
