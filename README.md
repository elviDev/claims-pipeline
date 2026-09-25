# Insurance Claims Data Pipeline

An end-to-end data and ML pipeline for insurance claims: raw data in, validated
tables, a fraud-risk model, an LLM that reads claim descriptions, and an API
that serves it all. Built to practise the stack used by data engineering teams
in insurance (Spark / Databricks, MLflow, Docker, CI/CD, LLMs).

> Work in progress. Built step by step, one working piece at a time.

## Roadmap

| Step | What | Status |
|------|------|--------|
| 1 | Synthetic claims and policies data, with realistic data quality problems | ✅ |
| 2 | Bronze / silver / gold layers in PySpark, with data quality checks | ✅ |
| 3 | Fraud-risk model, tracked with MLflow | ⏳ |
| 4 | FastAPI service that scores a claim, running in Docker | ⏳ |
| 5 | LLM step: extract damage type and urgency from the claim description | ⏳ |
| 6 | GitHub Actions: tests and Docker build on every push | ⏳ |
| 7 | Run the pipeline on Databricks | ⏳ |

## How the pipeline works

```
 raw CSV ──► BRONZE ──► SILVER ──────────► GOLD
             as-is,     typed, deduped,    claims_monthly  (reporting)
             all text   rules checked      fraud_features  (ML model)
                           │
                           └──► QUARANTINE  (bad rows + the rules they broke)
```

**Bronze** stores the raw data exactly as received, plus load time and source
file. Nothing is fixed here, so silver can always be rebuilt if a bug is found.

**Silver** converts types, removes duplicates and checks every row against data
quality rules. Problems with a safe fix are fixed (a missing channel becomes
`unknown`). Problems that would mean guessing, like a missing claim amount, go
to **quarantine** with the reason attached. Nothing is silently dropped.

**Quality gate**: if more than 5% of claims fail the rules, the run stops
before writing gold. A broken source file should never reach reports or models.

**Gold** builds tables for a specific use: a monthly summary for reporting, and
one row per claim with fraud signals for the model.

### Data quality rules

| Rule | Check | Action |
|------|-------|--------|
| `claim_amount_present` | amount is not missing | quarantine |
| `claim_amount_positive` | amount > 0 | quarantine |
| `incident_date_valid` | incident date is a real date | quarantine |
| `reported_after_incident` | report date ≥ incident date | quarantine |
| `policy_exists` | claim belongs to a known policy | quarantine |
| duplicate `claim_id` | same claim loaded twice | keep one |
| missing `channel` | not critical to the claim | fill with `unknown` |

Each run writes `data/quality_report.json`. On the default generated data, the
pipeline catches exactly the problems step 1 injected:

```
claims loaded:        20,200
duplicates removed:      200
quarantined:             498  (2.5%)
  missing amount         200
  negative amount        100
  reported too early     100
  unknown policy         100
clean claims in silver: 19,502
```

## Running it

### With Docker (recommended, works the same on Windows, Mac and Linux)

Spark needs Java, and on Windows it also needs extra Hadoop tools to write
files. Docker avoids all of that.

```bash
docker compose build
docker compose run --rm pipeline python src/generate_claims.py
docker compose run --rm pipeline                     # runs the pipeline
docker compose run --rm pipeline pytest              # runs the tests
```

Output lands in `data/` on your machine.

### Without Docker

Needs Python 3.10+ and Java 17 or 21.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt

python src/generate_claims.py
PYTHONPATH=src python -m pipeline.run_pipeline
pytest
```

## Project structure

```
src/
  generate_claims.py     step 1: synthetic data
  pipeline/
    spark.py             Spark session settings
    bronze.py            raw ingestion
    quality.py           rule engine: apply rules, split valid/quarantine, report
    silver.py            cleaning + the rules for policies and claims
    gold.py              reporting and feature tables
    run_pipeline.py      runs everything, applies the quality gate
tests/                   19 tests, one per rule plus end to end
```

## The data

**policies.csv**: one row per policy (product, region, customer age, start and end date, premium).

**claims.csv**: one row per claim (policy, incident and report dates, amount,
channel, free-text description, fraud label).

The fraud label follows real red flags: claims filed in the first month of a
policy, amounts far above normal for the product, and incidents reported late.
The raw data is intentionally dirty (missing values, negative amounts,
impossible dates, unknown policies, duplicates) so the pipeline has real
problems to catch.
