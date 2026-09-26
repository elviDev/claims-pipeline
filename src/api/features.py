"""
Turn raw claims into the features the model was trained on.

The claims system knows dates, amounts and the policy. It doesn't know
"days_since_policy_start" or "amount_vs_product_median": those were computed
by the Spark gold layer. So the API has to compute them itself, the same way.

This is where training/serving skew happens. If this code differs from the
Spark code even slightly (a date difference off by one, a different median,
different rounding), the model gets numbers it never saw in training. It
still returns a score, with no error, and the score is quietly wrong.

Two things protect against that:
  * the product medians come from the model's own MLflow run, not recomputed here
  * tests/test_features.py runs the same claims through Spark and through this
    code, and checks every feature matches exactly
"""

import pandas as pd

from model.train import FEATURES, NUMERIC

# The fields a raw claim has to include. These come straight from the claims
# and policy systems, no calculation needed.
RAW_FIELDS = [
    "product",
    "region",
    "channel",
    "customer_age",
    "claim_amount",
    "annual_premium",
    "policy_start_date",
    "incident_date",
    "report_date",
]


def build_features(claims: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """
    Raw claims in (one row each, columns RAW_FIELDS), model features out.

    Works on many rows at once, so /score and /score/batch use the same code.
    Each line mirrors a line in pipeline/silver.py or pipeline/gold.py; the
    comment says which.
    """
    # silver: F.lower(F.trim("product"))
    product = claims["product"].str.strip().str.lower()
    unknown = set(product) - set(medians)
    if unknown:
        # The request model already rejects unknown products. This is a second
        # guard in case this function gets called from somewhere else.
        raise ValueError(f"No median for product(s) {sorted(unknown)}: the model wasn't trained on them")

    # pd.to_datetime turns Python dates into pandas timestamps at midnight,
    # so subtracting two gives whole days, like Spark's datediff.
    start = pd.to_datetime(claims["policy_start_date"])
    incident = pd.to_datetime(claims["incident_date"])
    report = pd.to_datetime(claims["report_date"])
    amount = claims["claim_amount"].astype("float64")

    features = pd.DataFrame({
        "product": product,
        # silver: F.trim("region")
        "region": claims["region"].str.strip(),
        # silver: F.coalesce(F.lower(F.trim("channel")), F.lit("unknown"))
        "channel": claims["channel"].fillna("unknown").str.strip().str.lower(),
        "customer_age": claims["customer_age"],
        "claim_amount": amount,
        "annual_premium": claims["annual_premium"],
        # gold: F.datediff("incident_date", "start_date")
        "days_since_policy_start": (incident - start).dt.days,
        # gold: F.datediff("report_date", "incident_date")
        "report_delay_days": (report - incident).dt.days,
        # gold: F.round(claim_amount / F.median("claim_amount").over(by_product), 3)
        # The median comes from training, NOT from the claims in this request:
        # a batch of 3 claims has its own median, and it means nothing.
        "amount_vs_product_median": (amount / product.map(medians)).round(3),
        # gold: F.round(F.col("claim_amount") / F.col("annual_premium"), 3)
        "amount_vs_premium": (amount / claims["annual_premium"]).round(3),
    }, index=claims.index)

    # Same column order and types as training. The model's signature expects
    # every number as a double (see load_features in model/train.py).
    features[NUMERIC] = features[NUMERIC].astype("float64")
    return features[FEATURES]
