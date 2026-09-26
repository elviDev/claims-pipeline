"""
Tests for the scoring API (step 4).

The API loads a real model from a real (temporary) MLflow registry, trained
by the fixture in conftest.py. So these tests cover the whole path a request
takes in production: registry -> model + threshold + medians -> features -> score.
"""

import json

import mlflow
import pytest
from fastapi.testclient import TestClient
from openai import APITimeoutError

from api.main import DEFAULT_MODEL_URI, app
from llm.extract import LLMExtractor

NORMAL_CLAIM = {
    "claim_id": "C-NORMAL",
    "product": "auto",
    "region": "Madrid",
    "channel": "web",
    "customer_age": 42,
    "claim_amount": 1500.0,
    "annual_premium": 600.0,
    "policy_start_date": "2024-01-01",
    "incident_date": "2024-06-10",     # five months into the policy
    "report_date": "2024-06-12",       # reported two days later
}

# The three red flags the generator uses for fraud, all at once:
# claimed a week after buying the policy, ~3x the usual amount, reported a month late
RED_FLAG_CLAIM = {
    **NORMAL_CLAIM,
    "claim_id": "C-RISKY",
    "claim_amount": 6000.0,
    "incident_date": "2024-01-08",
    "report_date": "2024-02-10",
}


@pytest.fixture(scope="module")
def client(trained_registry):
    """
    Start the API against the test registry. `with TestClient(...)` runs the
    startup code (the lifespan), so the model really gets loaded, just like
    when uvicorn starts.
    """
    with pytest.MonkeyPatch.context() as env:
        env.setenv("MLFLOW_TRACKING_URI", trained_registry.tracking_uri)
        env.setenv("MODEL_URI", DEFAULT_MODEL_URI)
        # .env is loaded inside the container. Remove the key so the API uses
        # the fake extractor: tests must never call a real LLM.
        env.delenv("LLM_API_KEY", raising=False)
        with TestClient(app) as test_client:
            yield test_client
    mlflow.set_tracking_uri(None)


# --- Health -------------------------------------------------------------------

def test_health_says_which_model_is_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == "1"          # a fresh registry, so the champion is version 1
    assert 0 < body["threshold"] < 1
    assert body["extractor"] == "fake"


# --- Scoring ------------------------------------------------------------------

def test_normal_claim_gets_a_score(client):
    response = client.post("/score", json=NORMAL_CLAIM)
    assert response.status_code == 200

    body = response.json()
    assert body["claim_id"] == "C-NORMAL"
    assert 0 <= body["fraud_score"] <= 1
    assert body["flag_for_review"] == (body["fraud_score"] >= body["threshold"])


def test_red_flags_score_higher_than_a_normal_claim(client):
    """The model learned something real: the red flags push the score up."""
    normal = client.post("/score", json=NORMAL_CLAIM).json()
    risky = client.post("/score", json=RED_FLAG_CLAIM).json()
    assert risky["fraud_score"] > normal["fraud_score"]


def test_batch_gives_the_same_scores_as_one_by_one(client):
    batch = client.post("/score/batch", json=[NORMAL_CLAIM, RED_FLAG_CLAIM]).json()
    single = [client.post("/score", json=c).json() for c in (NORMAL_CLAIM, RED_FLAG_CLAIM)]
    assert batch == single                        # same order, same numbers


def test_empty_batch_returns_empty_list(client):
    assert client.post("/score/batch", json=[]).json() == []


def test_text_is_cleaned_like_silver(client):
    """Silver lowercases product and channel, so the API accepts "Auto" and " WEB " too."""
    messy = {**NORMAL_CLAIM, "product": "Auto", "channel": " WEB "}
    assert client.post("/score", json=messy).json() == client.post("/score", json=NORMAL_CLAIM).json()


def test_missing_channel_becomes_unknown(client):
    claim = {k: v for k, v in NORMAL_CLAIM.items() if k != "channel"}
    assert client.post("/score", json=claim).status_code == 200


# --- Bad input gets a clear 422, not a meaningless score ----------------------

@pytest.mark.parametrize("change, bad_field", [
    ({"claim_amount": -50}, "claim_amount"),                  # silver: claim_amount_positive
    ({"claim_amount": 0}, "claim_amount"),
    ({"annual_premium": 0}, "annual_premium"),                # silver: premium_positive
    ({"product": "pet"}, "product"),                          # the model never saw it
    ({"region": "Narnia"}, "region"),
    ({"incident_date": "2024-13-45"}, "incident_date"),       # not a real date
    ({"claim_amount": None}, "claim_amount"),                 # silver: claim_amount_present
])
def test_bad_field_is_rejected(client, change, bad_field):
    response = client.post("/score", json={**NORMAL_CLAIM, **change})
    assert response.status_code == 422
    assert bad_field in response.json()["detail"][0]["loc"]


def test_report_before_incident_is_rejected(client):
    """silver: reported_after_incident."""
    claim = {**NORMAL_CLAIM, "incident_date": "2024-06-10", "report_date": "2024-06-01"}
    response = client.post("/score", json=claim)
    assert response.status_code == 422
    assert "report_date" in response.json()["detail"][0]["msg"]


# --- Logging ------------------------------------------------------------------

def test_each_prediction_is_logged_as_json(client, caplog):
    with caplog.at_level("INFO", logger="api.predictions"):
        client.post("/score", json=NORMAL_CLAIM)

    logged = json.loads(caplog.records[-1].getMessage())
    assert logged["claim_id"] == "C-NORMAL"
    assert logged["model_version"] == "1"
    assert set(logged) >= {"fraud_score", "flagged", "latency_ms"}


# --- Triage: fraud score + fields from the description ------------------------

TRIAGE_CLAIM = {**NORMAL_CLAIM, "description": "Burst pipe in the bathroom flooded the hallway."}


def test_triage_returns_score_and_extracted_fields(client):
    response = client.post("/claims/triage", json=TRIAGE_CLAIM)
    assert response.status_code == 200

    body = response.json()
    # The fraud part is exactly what /score gives for the same claim
    assert body["fraud"] == client.post("/score", json=NORMAL_CLAIM).json()
    assert body["extraction"]["damage_type"] == "water"
    assert body["extraction"]["urgency"] == "high"


def test_triage_says_when_no_llm_is_used(client):
    body = client.post("/claims/triage", json=TRIAGE_CLAIM).json()
    assert body["extraction"]["extractor"] == "fake"
    assert "keyword rules" in body["note"]


def test_triage_needs_a_description(client):
    assert client.post("/claims/triage", json=NORMAL_CLAIM).status_code == 422
    assert client.post("/claims/triage", json={**NORMAL_CLAIM, "description": ""}).status_code == 422


class TimingOutClient:
    """Stands in for openai.OpenAI(); every call times out, like an overloaded LLM server."""
    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **request):
        raise APITimeoutError(request=None)


def test_llm_failure_still_returns_the_fraud_score(client):
    """If the LLM is down, the claim still gets scored; the extraction is marked for a person."""
    original = app.state.extractor
    app.state.extractor = LLMExtractor(model="down", client=TimingOutClient())
    try:
        body = client.post("/claims/triage", json=TRIAGE_CLAIM).json()
    finally:
        app.state.extractor = original

    assert 0 <= body["fraud"]["fraud_score"] <= 1
    assert body["extraction"]["needs_manual_review"] is True
    assert body["note"] is None                 # it was a real LLM, it just failed
