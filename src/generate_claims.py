"""
Step 1: Generate a synthetic insurance dataset.

Produces two CSV files that act as the "raw" data landing in our system,
the way exports from a policy system and a claims system would:

    data/raw/policies.csv
    data/raw/claims.csv

The data is deliberately a bit dirty (missing values, duplicates, negative
amounts, claims dated before the policy started). That's on purpose: the
data quality step later in the pipeline needs real problems to catch.

Run it:
    python src/generate_claims.py --n-policies 5000 --n-claims 20000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reference values
# ---------------------------------------------------------------------------

PRODUCTS = ["auto", "home", "travel", "health"]
REGIONS = ["Catalonia", "Madrid", "Andalusia", "Valencia", "Basque Country", "Galicia"]
CHANNELS = ["web", "phone", "agent", "app"]

# Free-text templates per product. The LLM step later will read these
# descriptions and pull out structured fields (damage type, urgency).
DESCRIPTIONS = {
    "auto": [
        "Rear-ended at a traffic light, bumper and trunk damaged.",
        "Hit a pole while parking, front left headlight broken.",
        "Windshield cracked by a stone on the highway.",
        "Car was broken into overnight, window smashed and radio stolen.",
        "Collision at a roundabout, other driver did not stop. Door dented.",
    ],
    "home": [
        "Water leak from the upstairs neighbour, kitchen ceiling stained.",
        "Burst pipe in the bathroom flooded the hallway.",
        "Storm blew tiles off the roof, rain coming in to the bedroom.",
        "Break-in while on holiday, laptop and jewellery missing.",
        "Small kitchen fire, cabinets and extractor fan damaged.",
    ],
    "travel": [
        "Luggage lost on connecting flight, still not returned after 5 days.",
        "Flight cancelled, had to pay for an extra hotel night.",
        "Phone stolen at the train station during the trip.",
        "Needed a doctor abroad for a stomach infection.",
    ],
    "health": [
        "Emergency room visit after a fall, wrist X-ray and cast.",
        "Physiotherapy sessions after knee surgery.",
        "Dental emergency, broken tooth needed a crown.",
        "Specialist consultation and blood tests.",
    ],
}

# Typical claim size per product (mean of a lognormal, in EUR)
TYPICAL_AMOUNT = {"auto": 1800, "home": 2500, "travel": 600, "health": 900}


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_policies(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """One row per insurance policy."""
    start = pd.Timestamp("2021-01-01")
    start_offsets = rng.integers(0, 3 * 365, size=n)

    policies = pd.DataFrame({
        "policy_id": [f"POL{i:06d}" for i in range(1, n + 1)],
        "product": rng.choice(PRODUCTS, size=n, p=[0.4, 0.3, 0.15, 0.15]),
        "region": rng.choice(REGIONS, size=n),
        "customer_age": rng.integers(18, 85, size=n),
        "start_date": start + pd.to_timedelta(start_offsets, unit="D"),
        "annual_premium": rng.normal(600, 180, size=n).clip(120).round(2),
    })
    policies["end_date"] = policies["start_date"] + pd.DateOffset(years=1)
    return policies


def generate_claims(policies: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """One row per claim, linked to a policy."""
    pol = policies.sample(n=n, replace=True, random_state=rng.integers(1e9)).reset_index(drop=True)

    # Claim happens somewhere inside the policy year
    days_into_policy = rng.integers(0, 365, size=n)
    incident_date = pol["start_date"] + pd.to_timedelta(days_into_policy, unit="D")
    report_delay = rng.exponential(5, size=n).astype(int)
    report_date = incident_date + pd.to_timedelta(report_delay, unit="D")

    typical = pol["product"].map(TYPICAL_AMOUNT).to_numpy()
    amount = rng.lognormal(mean=np.log(typical), sigma=0.6).round(2)

    descriptions = [rng.choice(DESCRIPTIONS[p]) for p in pol["product"]]

    claims = pd.DataFrame({
        "claim_id": [f"CLM{i:07d}" for i in range(1, n + 1)],
        "policy_id": pol["policy_id"],
        "incident_date": incident_date,
        "report_date": report_date,
        "claim_amount": amount,
        "channel": rng.choice(CHANNELS, size=n),
        "description": descriptions,
    })

    # --- Fraud label -------------------------------------------------------
    # Real fraud has patterns. We build a risk score from a few known red
    # flags so a model can actually learn something:
    #   * claim filed very early in the policy (people buy a policy, then claim)
    #   * amount much larger than normal for that product
    #   * incident reported long after it happened
    early = (days_into_policy < 30).astype(float)
    big = (amount > 2.5 * typical).astype(float)
    late = (report_delay > 20).astype(float)

    logit = -4.0 + 1.8 * early + 2.0 * big + 1.2 * late
    prob = 1 / (1 + np.exp(-logit))
    claims["is_fraud"] = (rng.random(n) < prob).astype(int)

    return claims


def add_dirty_data(claims: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject the kind of problems real source systems have."""
    claims = claims.copy()
    n = len(claims)

    def pick(frac):
        return rng.choice(n, size=int(n * frac), replace=False)

    # 1% missing amounts, 1% missing channel
    claims.loc[pick(0.01), "claim_amount"] = np.nan
    claims.loc[pick(0.01), "channel"] = None

    # 0.5% negative amounts (a typo or a refund recorded in the wrong place)
    idx = pick(0.005)
    claims.loc[idx, "claim_amount"] = -claims.loc[idx, "claim_amount"].abs()

    # 0.5% reported before the incident happened (impossible)
    idx = pick(0.005)
    claims.loc[idx, "report_date"] = claims.loc[idx, "incident_date"] - pd.Timedelta(days=10)

    # 0.5% point to a policy that doesn't exist
    idx = pick(0.005)
    claims.loc[idx, "policy_id"] = "POL999999"

    # 1% exact duplicate rows (the export ran twice)
    dupes = claims.iloc[pick(0.01)]
    claims = pd.concat([claims, dupes], ignore_index=True)

    # Shuffle so the problems aren't all at the bottom
    return claims.sample(frac=1, random_state=rng.integers(1e9)).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(n_policies: int, n_claims: int, out_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = generate_policies(n_policies, rng)
    claims = generate_claims(policies, n_claims, rng)
    claims = add_dirty_data(claims, rng)

    policies.to_csv(out_dir / "policies.csv", index=False)
    claims.to_csv(out_dir / "claims.csv", index=False)

    print(f"Wrote {len(policies):,} policies and {len(claims):,} claims to {out_dir}/")
    print(f"Fraud rate: {claims['is_fraud'].mean():.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic insurance data.")
    parser.add_argument("--n-policies", type=int, default=5000)
    parser.add_argument("--n-claims", type=int, default=20000)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args.n_policies, args.n_claims, args.out_dir, args.seed)
