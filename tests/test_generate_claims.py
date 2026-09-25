import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_claims import add_dirty_data, generate_claims, generate_policies  # noqa: E402


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    policies = generate_policies(500, rng)
    claims = generate_claims(policies, 2000, rng)
    return policies, claims, rng


def test_policy_ids_are_unique(data):
    policies, _, _ = data
    assert policies["policy_id"].is_unique


def test_clean_claims_link_to_real_policies(data):
    policies, claims, _ = data
    assert claims["policy_id"].isin(policies["policy_id"]).all()


def test_clean_claims_have_positive_amounts(data):
    _, claims, _ = data
    assert (claims["claim_amount"] > 0).all()


def test_fraud_rate_is_realistic(data):
    # Real fraud rates in insurance sit roughly between 1% and 15%
    _, claims, _ = data
    assert 0.01 < claims["is_fraud"].mean() < 0.15


def test_dirty_data_actually_adds_problems(data):
    _, claims, rng = data
    dirty = add_dirty_data(claims, rng)
    assert len(dirty) > len(claims)                      # duplicates added
    assert dirty["claim_amount"].isna().any()            # missing values
    assert (dirty["claim_amount"] < 0).any()             # negative amounts
    assert (dirty["report_date"] < dirty["incident_date"]).any()  # impossible dates


def test_same_seed_gives_same_data():
    a = generate_policies(100, np.random.default_rng(7))
    b = generate_policies(100, np.random.default_rng(7))
    assert a.equals(b)
