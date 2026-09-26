"""
Tests for the API's feature code, above all the training/serving skew test.

The model learned from features computed by Spark (pipeline/gold.py). The API
computes the same features with pandas (api/features.py). If the two ever
disagree, the model gets inputs it never trained on and its scores are quietly
wrong. These tests make that disagreement fail loudly instead.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from api.features import RAW_FIELDS, build_features
from generate_claims import add_dirty_data, generate_claims, generate_policies
from model.train import FEATURES, NUMERIC, product_medians
from pipeline.bronze import read_raw_csv
from pipeline.gold import build_fraud_features
from pipeline.silver import build_silver_claims, build_silver_policies

MEDIANS = {"auto": 1800.0, "home": 2500.0, "travel": 600.0, "health": 900.0}


def raw_claim(**overrides) -> dict:
    claim = dict(
        product="auto", region="Madrid", channel="web", customer_age=40.0,
        claim_amount=3600.0, annual_premium=600.0,
        policy_start_date=date(2024, 1, 1), incident_date=date(2024, 1, 11), report_date=date(2024, 1, 14),
    )
    claim.update(overrides)
    return claim


# --- Small cases, worked out by hand ------------------------------------------

def test_features_for_one_claim():
    row = build_features(pd.DataFrame([raw_claim()]), MEDIANS).iloc[0]

    assert list(row.index) == FEATURES
    assert row["days_since_policy_start"] == 10     # 1 Jan -> 11 Jan
    assert row["report_delay_days"] == 3            # 11 Jan -> 14 Jan
    assert row["amount_vs_product_median"] == 2.0   # 3600 / 1800
    assert row["amount_vs_premium"] == 6.0          # 3600 / 600


def test_text_is_cleaned_like_silver():
    row = build_features(pd.DataFrame([raw_claim(product=" Auto ", channel=None)]), MEDIANS).iloc[0]
    assert row["product"] == "auto"
    assert row["channel"] == "unknown"


def test_median_comes_from_training_not_from_the_batch():
    """Two auto claims in one batch: each is compared to the TRAINING median, not to each other."""
    batch = pd.DataFrame([raw_claim(claim_amount=900.0), raw_claim(claim_amount=9000.0)])
    ratios = build_features(batch, MEDIANS)["amount_vs_product_median"].tolist()
    assert ratios == [0.5, 5.0]


def test_unknown_product_is_refused():
    with pytest.raises(ValueError, match="pet"):
        build_features(pd.DataFrame([raw_claim(product="pet")]), MEDIANS)


# --- The skew test ------------------------------------------------------------

def test_api_features_match_spark_gold_features(spark, tmp_path):
    """
    Same claims, two code paths, identical features.

    Spark path: raw CSV -> bronze -> silver -> gold, exactly like the pipeline.
    API path:   the same raw claims, joined to their policy, through build_features.
    """
    rng = np.random.default_rng(11)
    policies = generate_policies(200, rng)
    # Dirty data on purpose: missing channels must become "unknown" in both paths
    claims = add_dirty_data(generate_claims(policies, 1500, rng), rng)
    policies.to_csv(tmp_path / "policies.csv", index=False)
    claims.to_csv(tmp_path / "claims.csv", index=False)

    # Spark path
    silver_policies, _, _ = build_silver_policies(read_raw_csv(spark, str(tmp_path / "policies.csv")))
    silver_claims, _, _ = build_silver_claims(read_raw_csv(spark, str(tmp_path / "claims.csv")), silver_policies)
    gold = build_fraud_features(silver_claims, silver_policies).toPandas()

    # API path, on the claims that made it to gold (the API rejects the bad ones
    # with a 422 instead). Medians the way train.py saves them.
    raw = (
        claims.drop_duplicates("claim_id")
        .merge(policies.rename(columns={"start_date": "policy_start_date"}), on="policy_id")
        .set_index("claim_id")
        .loc[gold["claim_id"]]
    )
    for col in ["policy_start_date", "incident_date", "report_date"]:
        raw[col] = raw[col].dt.date                 # the API receives plain dates
    api = build_features(raw[RAW_FIELDS], product_medians(gold))

    spark_side = gold.set_index("claim_id")[FEATURES]
    spark_side[NUMERIC] = spark_side[NUMERIC].astype("float64")

    assert len(api) > 1000
    assert "unknown" in set(api["channel"])         # the missing-channel path really ran
    for column in FEATURES:
        mismatches = (api[column] != spark_side[column]).sum()
        assert mismatches == 0, f"{column}: {mismatches} claims differ between Spark and the API"
