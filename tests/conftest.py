import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pytest

from generate_claims import generate_claims, generate_policies
from model.train import run
from pipeline.bronze import read_raw_csv
from pipeline.gold import build_fraud_features
from pipeline.silver import build_silver_claims, build_silver_policies
from pipeline.spark import get_spark


@pytest.fixture(scope="session")
def spark():
    """One Spark session shared by all tests (starting Spark takes a few seconds)."""
    session = get_spark("claims-pipeline-tests")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def features_path(spark, tmp_path_factory) -> Path:
    """
    Generate a small dataset and run it through silver and gold. Returns the gold folder.
    Shared by the model and API tests, so the Spark part runs once per test session.
    """
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


@dataclass
class TrainedRegistry:
    tracking_uri: str   # point MLflow here to see the test runs and registered model
    metrics: dict       # test-set metrics per model, as returned by model.train.run


@pytest.fixture(scope="session")
def trained_registry(features_path, tmp_path_factory) -> TrainedRegistry:
    """
    Run the full training script once, into a temporary MLflow registry.

    MLflow writes mlflow.db and mlruns/ into the current folder, so we move to
    a temp folder first. Otherwise every test run would add fake runs to the
    real registry. The model files are stored with their full path, so they
    can still be loaded after we move back.
    """
    workdir = tmp_path_factory.mktemp("mlflow")
    tracking_uri = f"sqlite:///{workdir}/mlflow.db"
    old_cwd = os.getcwd()
    os.chdir(workdir)
    try:
        metrics = run(features_path, review_rate=0.1, tracking_uri=tracking_uri)
    finally:
        os.chdir(old_cwd)
        mlflow.set_tracking_uri(None)  # each test module points MLflow where it needs
    return TrainedRegistry(tracking_uri, metrics)
