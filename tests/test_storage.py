"""
Tests for where the pipeline stores each layer (step 7), and for a full run.

Table mode is what runs on Databricks. We can't run Unity Catalog here, so
those tests check the names and the Spark calls with stand-ins that record
what they were asked to do.
"""

import numpy as np
import pytest

from generate_claims import add_dirty_data, generate_claims, generate_policies
from pipeline.run_pipeline import run
from pipeline.storage import Storage


# --- Naming -------------------------------------------------------------------

@pytest.mark.parametrize("layer, name, table", [
    ("bronze", "claims", "workspace.claims.bronze_claims"),
    ("silver", "policies", "workspace.claims.silver_policies"),
    ("quarantine", "claims", "workspace.claims.quarantine_claims"),
    ("gold", "fraud_features", "workspace.claims.gold_fraud_features"),
    ("gold", "claims_monthly", "workspace.claims.gold_claims_monthly"),
])
def test_table_names_follow_the_medallion_convention(layer, name, table):
    assert Storage("table", "workspace.claims").location(layer, name) == table


def test_path_mode_keeps_the_local_folders():
    assert Storage("path", "data").location("silver", "claims") == "data/silver/claims"
    assert Storage("path", "/app/data/").location("gold", "fraud_features") == "/app/data/gold/fraud_features"


@pytest.mark.parametrize("base", ["workspace", "workspace.claims.extra"])
def test_table_mode_needs_catalog_dot_schema(base):
    with pytest.raises(ValueError, match="catalog.schema"):
        Storage("table", base)


# --- Table mode, with stand-ins for Spark ---------------------------------------

class Recorder:
    """Records every method call made on it, and returns itself so calls can be chained."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, method):
        def record(*args):
            self.calls.append((method, *args))
            return self
        return record


def test_table_mode_writes_with_save_as_table():
    writer = Recorder()
    df = type("StandInDataFrame", (), {"write": writer})()

    Storage("table", "workspace.claims").write(df, "silver", "claims")

    assert ("mode", "overwrite") in writer.calls
    assert ("saveAsTable", "workspace.claims.silver_claims") in writer.calls


def test_table_mode_reads_the_table_by_name():
    spark = Recorder()
    Storage("table", "workspace.claims").read(spark, "gold", "fraud_features")
    assert spark.calls == [("table", "workspace.claims.gold_fraud_features")]


# --- A full run, in path mode ---------------------------------------------------

@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(3)
    policies = generate_policies(200, rng)
    add_dirty_data(generate_claims(policies, 1000, rng), rng).to_csv(raw / "claims.csv", index=False)
    policies.to_csv(raw / "policies.csv", index=False)
    return raw


def test_full_run_writes_every_layer(spark, raw_dir, tmp_path):
    storage = Storage("path", str(tmp_path))
    report = run(storage, str(raw_dir), spark)

    silver = storage.read(spark, "silver", "claims").count()
    quarantine = storage.read(spark, "quarantine", "claims").count()
    features = storage.read(spark, "gold", "fraud_features").count()

    assert report["claims"]["valid_rows"] == silver
    assert report["claims"]["quarantined_rows"] == quarantine
    assert quarantine > 0                          # the dirty data got caught
    assert features == silver                      # one feature row per clean claim
    assert storage.read(spark, "gold", "claims_monthly").count() > 0


def test_running_twice_gives_the_same_result(spark, raw_dir, tmp_path):
    """Each layer is overwritten, not appended to: a rerun must never duplicate rows."""
    storage = Storage("path", str(tmp_path))
    first = run(storage, str(raw_dir), spark)
    second = run(storage, str(raw_dir), spark)

    assert first == second
    assert storage.read(spark, "silver", "claims").count() == second["claims"]["valid_rows"]
