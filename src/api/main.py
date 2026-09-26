"""
Step 4: an API that scores a new claim for fraud risk.

A claims system sends a raw claim (dates, amount, policy details) and gets
back a fraud score and whether the claim should go to a human for review.

Run it:
    uvicorn api.main:app --port 8000          (then open http://localhost:8000/docs)

Settings, from environment variables:
    MODEL_URI            which model to serve    (default models:/claims-fraud-model@champion)
    MLFLOW_TRACKING_URI  where the registry is   (default sqlite:///mlflow.db)
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, Request
from mlflow import MlflowClient
from pydantic import BaseModel, Field, field_validator, model_validator

from api.features import build_features
from model.train import CHAMPION_ALIAS, DEFAULT_TRACKING_URI, MEDIANS_ARTIFACT, REGISTERED_MODEL

DEFAULT_MODEL_URI = f"models:/{REGISTERED_MODEL}@{CHAMPION_ALIAS}"

# One JSON line per prediction. Nobody reads these by eye: they're the raw
# material for monitoring later (are scores drifting? is the flag rate still
# ~5%? is it getting slower?). JSON so a log tool can parse them.
prediction_log = logging.getLogger("api.predictions")
prediction_log.setLevel(logging.INFO)
if not prediction_log.handlers:
    prediction_log.addHandler(logging.StreamHandler())


# ---------------------------------------------------------------------------
# Request and response
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    """
    A raw claim, as the claims system knows it.

    The checks here are the silver layer's quality rules. In the pipeline, a
    bad row goes to quarantine; here, a bad request gets a 422 with the reason.
    Either way, bad data never reaches the model. A score for a claim with a
    negative amount would look like an answer, but it would be meaningless.
    """
    claim_id: str = Field(min_length=1, examples=["CLM0000123"])
    # Only values the model was trained on. An unknown product has no median,
    # and the model has never seen it: better to refuse than to guess.
    product: Literal["auto", "home", "travel", "health"]
    region: Literal["Catalonia", "Madrid", "Andalusia", "Valencia", "Basque Country", "Galicia"]
    # "unknown" is what silver fills in for a missing channel, so it's a value the model knows
    channel: Literal["web", "phone", "agent", "app", "unknown"] = "unknown"
    customer_age: float = Field(ge=18, le=120, examples=[42])
    claim_amount: float = Field(gt=0, examples=[1850.0])            # silver: claim_amount_positive
    annual_premium: float = Field(gt=0, examples=[600.0])           # silver: premium_positive
    policy_start_date: date = Field(examples=["2024-01-01"])
    incident_date: date = Field(examples=["2024-05-10"])
    report_date: date = Field(examples=["2024-05-12"])

    # Silver lowercases product and channel, so "Auto " is fine there. Same here.
    @field_validator("product", "channel", mode="before")
    @classmethod
    def lowercase(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    # silver: reported_after_incident
    @model_validator(mode="after")
    def reported_after_incident(self):
        if self.report_date < self.incident_date:
            raise ValueError("report_date can't be before incident_date")
        return self


class Score(BaseModel):
    claim_id: str
    fraud_score: float = Field(description="0 to 1, higher is more suspicious. A ranking, not a probability.")
    flag_for_review: bool = Field(description="True if fraud_score >= threshold")
    threshold: float
    model_version: str


# ---------------------------------------------------------------------------
# The model, loaded once
# ---------------------------------------------------------------------------

@dataclass
class Scorer:
    """Everything that belongs to one model version, loaded together."""
    model: mlflow.pyfunc.PyFuncModel
    version: str
    model_type: str
    threshold: float
    medians: dict[str, float]

    def score(self, claims: list[Claim]) -> list[Score]:
        raw = pd.DataFrame([c.model_dump() for c in claims])
        features = build_features(raw, self.medians)
        # predict_proba gives [P(honest), P(fraud)] per claim; we want the second
        scores = self.model.predict(features)[:, 1]
        return [
            Score(
                claim_id=claim.claim_id,
                fraud_score=round(float(s), 4),
                flag_for_review=bool(s >= self.threshold),
                threshold=self.threshold,
                model_version=self.version,
            )
            for claim, s in zip(claims, scores)
        ]


def load_scorer(model_uri: str) -> Scorer:
    """
    Resolve "models:/claims-fraud-model@champion" to a version number, then
    load the model, threshold and medians all from THAT version.

    Why resolve first: if someone moves the alias while we're loading, we could
    otherwise end up with version 3's model and version 4's threshold.
    """
    if not (model_uri.startswith("models:/") and "@" in model_uri):
        raise ValueError(f"MODEL_URI must look like models:/<name>@<alias>, got {model_uri!r}")
    name, alias = model_uri.removeprefix("models:/").split("@", 1)

    client = MlflowClient()
    version = client.get_model_version_by_alias(name, alias)

    return Scorer(
        model=mlflow.pyfunc.load_model(f"models:/{name}/{version.version}"),
        version=str(version.version),
        model_type=version.tags.get("model_type", "unknown"),
        threshold=float(version.tags["threshold"]),
        # Saved in the same run as the model by train.py, so they always match it
        medians=mlflow.artifacts.load_dict(f"runs:/{version.run_id}/{MEDIANS_ARTIFACT}"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts. If the model can't be loaded, the server
    doesn't start at all. That's on purpose: an API that is "up" but can't
    score would pass a basic check and then fail every real request.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    app.state.scorer = load_scorer(os.getenv("MODEL_URI", DEFAULT_MODEL_URI))
    yield


app = FastAPI(
    title="Claims fraud scoring",
    description="Scores an insurance claim for fraud risk and says whether a human should review it.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def score_and_log(request: Request, claims: list[Claim]) -> list[Score]:
    start = time.perf_counter()
    results = request.app.state.scorer.score(claims)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    for r in results:
        prediction_log.info(json.dumps({
            "claim_id": r.claim_id,
            "fraud_score": r.fraud_score,
            "flagged": r.flag_for_review,
            "model_version": r.model_version,
            "latency_ms": latency_ms,       # for a batch, the time for the whole batch
            "batch_size": len(claims),
        }))
    return results


@app.get("/health")
def health(request: Request) -> dict:
    """
    Is the service up, and which model is it serving? A load balancer or a
    Kubernetes probe calls this; so does a person checking what's deployed.
    """
    scorer: Scorer = request.app.state.scorer
    return {
        "status": "ok",
        "model": REGISTERED_MODEL,
        "model_version": scorer.version,
        "model_type": scorer.model_type,
        "threshold": scorer.threshold,
    }


@app.post("/score")
def score(claim: Claim, request: Request) -> Score:
    """Score one claim. For a claims system checking each claim as it comes in."""
    return score_and_log(request, [claim])[0]


@app.post("/score/batch")
def score_batch(claims: list[Claim], request: Request) -> list[Score]:
    """
    Score many claims in one call, e.g. a nightly job. One model call for the
    whole list is much faster than one HTTP request per claim.
    """
    if not claims:
        return []
    return score_and_log(request, claims)
