"""
Tests for the Spark pipeline.

Each test builds a tiny DataFrame by hand, with exactly the problem we want
to check, so when a test fails it's obvious why.
"""

import numpy as np
import pytest
from pyspark.sql import functions as F

from generate_claims import add_dirty_data, generate_claims, generate_policies
from pipeline.bronze import read_raw_csv
from pipeline.gold import build_claims_monthly, build_fraud_features
from pipeline.silver import build_silver_claims, build_silver_policies

POLICY_COLS = ["policy_id", "product", "region", "customer_age", "start_date", "end_date", "annual_premium"]
CLAIM_COLS = ["claim_id", "policy_id", "incident_date", "report_date", "claim_amount", "channel", "description", "is_fraud"]


def string_schema(cols):
    """Bronze has only text columns. Saying so explicitly lets us pass None values."""
    return ", ".join(f"{c} STRING" for c in cols)


@pytest.fixture
def bronze_policies(spark):
    # Bronze is all strings, just like the raw CSV
    rows = [
        ("POL1", "auto", "Madrid", "40", "2023-01-01", "2024-01-01", "500.0"),
        ("POL2", "home", "Catalonia", "55", "2023-03-01", "2024-03-01", "700.0"),
    ]
    return spark.createDataFrame(rows, string_schema(POLICY_COLS))


def make_claims(spark, rows):
    return spark.createDataFrame(rows, string_schema(CLAIM_COLS))


def good_claim(claim_id="C1", **overrides):
    row = dict(
        claim_id=claim_id, policy_id="POL1", incident_date="2023-02-10", report_date="2023-02-12",
        claim_amount="1200.50", channel="web", description="Bumper damaged.", is_fraud="0",
    )
    row.update(overrides)
    return tuple(row[c] for c in CLAIM_COLS)


def run_silver(spark, bronze_policies, claim_rows):
    policies, _, _ = build_silver_policies(bronze_policies)
    return build_silver_claims(make_claims(spark, claim_rows), policies)


# --- Silver: each quality rule catches its problem ---------------------------

def test_good_claim_passes(spark, bronze_policies):
    valid, quarantine, _ = run_silver(spark, bronze_policies, [good_claim()])
    assert valid.count() == 1
    assert quarantine.count() == 0


@pytest.mark.parametrize("overrides, expected_rule", [
    ({"claim_amount": None}, "claim_amount_present"),
    ({"claim_amount": "-50"}, "claim_amount_positive"),
    ({"claim_amount": "abc"}, "claim_amount_present"),        # text that isn't a number becomes NULL
    ({"incident_date": "not-a-date"}, "incident_date_valid"),
    ({"report_date": "2023-02-01"}, "reported_after_incident"),
    ({"policy_id": "POL999"}, "policy_exists"),
])
def test_bad_claim_is_quarantined_with_reason(spark, bronze_policies, overrides, expected_rule):
    valid, quarantine, _ = run_silver(spark, bronze_policies, [good_claim(**overrides)])
    assert valid.count() == 0
    failed = quarantine.first()["_failed_rules"]
    assert expected_rule in failed


def test_duplicates_are_removed(spark, bronze_policies):
    valid, _, _ = run_silver(spark, bronze_policies, [good_claim("C1"), good_claim("C1")])
    assert valid.count() == 1


def test_missing_channel_is_fixed_not_quarantined(spark, bronze_policies):
    valid, quarantine, _ = run_silver(spark, bronze_policies, [good_claim(channel=None)])
    assert quarantine.count() == 0
    assert valid.first()["channel"] == "unknown"


def test_types_are_converted(spark, bronze_policies):
    valid, _, _ = run_silver(spark, bronze_policies, [good_claim()])
    types = dict(valid.dtypes)
    assert types["claim_amount"] == "double"
    assert types["incident_date"] == "date"
    assert types["is_fraud"] == "int"


# --- Gold ---------------------------------------------------------------------

def test_fraud_features(spark, bronze_policies):
    claims, _, _ = run_silver(spark, bronze_policies, [good_claim()])
    policies, _, _ = build_silver_policies(bronze_policies)
    row = build_fraud_features(claims, policies).first()

    assert row["days_since_policy_start"] == 40      # 2023-01-01 -> 2023-02-10
    assert row["report_delay_days"] == 2
    assert row["amount_vs_product_median"] == 1.0    # only claim, so it IS the median
    assert row["amount_vs_premium"] == pytest.approx(1200.5 / 500, abs=0.001)


def test_claims_monthly_totals(spark, bronze_policies):
    rows = [good_claim("C1", claim_amount="100"), good_claim("C2", claim_amount="300", is_fraud="1")]
    claims, _, _ = run_silver(spark, bronze_policies, rows)
    policies, _, _ = build_silver_policies(bronze_policies)
    row = build_claims_monthly(claims, policies).first()

    assert row["n_claims"] == 2
    assert row["total_amount"] == 400
    assert row["fraud_rate"] == 0.5


# --- Whole thing on generated data -------------------------------------------

def test_pipeline_on_generated_data(spark, tmp_path):
    """
    End to end on generated dirty data, through the real CSV reader.
    Every claim must end up in exactly one place: silver or quarantine. Nothing lost.
    """
    rng = np.random.default_rng(1)
    pol = generate_policies(300, rng)
    clm = add_dirty_data(generate_claims(pol, 1000, rng), rng)
    pol.to_csv(tmp_path / "policies.csv", index=False)
    clm.to_csv(tmp_path / "claims.csv", index=False)

    bronze_pol = read_raw_csv(spark, str(tmp_path / "policies.csv"))
    bronze_clm = read_raw_csv(spark, str(tmp_path / "claims.csv"))
    assert bronze_clm.count() == len(clm)                  # bronze keeps every raw row
    assert "_source_file" in bronze_clm.columns

    policies, _, _ = build_silver_policies(bronze_pol)
    valid, quarantine, _ = build_silver_claims(bronze_clm, policies)

    unique_claims = clm["claim_id"].nunique()
    assert valid.count() + quarantine.count() == unique_claims
    assert quarantine.count() > 0                          # the dirty data got caught
    assert valid.filter(F.col("claim_amount") <= 0).count() == 0
