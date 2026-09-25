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
| 3 | Fraud-risk model, tracked with MLflow | ✅ |
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

## Fraud model

`src/model/train.py` trains on `gold/fraud_features` (one row per claim) and
records everything in MLflow.

**Two models.** A logistic regression baseline, and a gradient-boosted tree
model. The baseline is there to answer "is the fancier model actually better?"
Both are scikit-learn Pipelines that include the data prep (one-hot encoding of
product, region and channel), so the saved model takes a raw claim and does its
own encoding. Class imbalance is handled with `class_weight="balanced"`.

**Choosing which claims to flag.** Investigators can only review so many
claims. The threshold is set so the model flags the riskiest 5% of claims
(`--review-rate`), measured on the training data, never the test data.

**Metrics.** About 3% of claims are fraud, so a "model" that never flags
anything is 97% accurate and useless. The main metric is PR AUC (average
precision), which measures how well fraud rises to the top of the list. A random
ranking scores about 0.03.

Results on the held-out test set (3,901 claims, 127 of them fraud):

| Model | ROC AUC | PR AUC | Precision | Recall | Accuracy |
|-------|---------|--------|-----------|--------|----------|
| Logistic regression (baseline) | 0.646 | 0.081 | 10.4% | 16.5% | 92.6% |
| **Gradient boosting** (registered) | 0.637 | **0.102** | **13.9%** | **21.3%** | 93.1% |
| Never flag anything | | 0.033 | | 0% | 96.7% |

Reviewing 5% of claims, the tree model catches about 21% of fraud, roughly 4x
better than picking claims at random. When it flags a claim, it's right 14% of
the time, against a 3.3% base rate.

**Why the scores aren't higher.** The generator labels a claim as fraud from
three red flags plus random chance, and 56% of fraud cases have no red flag at
all. Scoring the test set with the exact formula that created the labels gives
ROC AUC 0.656 and PR AUC 0.089, so the models are already at the ceiling this
data allows.

**Known shortcut.** `amount_vs_product_median` is computed over all claims
before the train/test split, so the test set has a small influence on a
training feature. Fine for a demo; in production it would be computed from
training data only.

**What MLflow records per run:** model settings, the review rate, the dataset
(with a fingerprint), the git commit, all metrics, the confusion matrix as an
image, and the model with its input signature and an example. The best model by
PR AUC is registered as `claims-fraud-model` with the alias `champion`, tagged
with its threshold. The API in step 4 loads `models:/claims-fraud-model@champion`,
so deploying a new model means moving the alias, not changing code.

Run records and the registry are stored in `mlflow.db` (SQLite), model files in
`mlruns/`. Both are git-ignored and rebuilt by running the training script.

## Running it

### With Docker (recommended, works the same on Windows, Mac and Linux)

Spark needs Java, and on Windows it also needs extra Hadoop tools to write
files. Docker avoids all of that.

```bash
docker compose build
docker compose run --rm pipeline python src/generate_claims.py
docker compose run --rm pipeline                     # runs the pipeline
docker compose run --rm pipeline python -m model.train   # trains and registers the model
docker compose run --rm pipeline pytest              # runs the tests
docker compose up mlflow                             # MLflow UI on http://localhost:5000
```

Output lands in `data/`, `mlflow.db` and `mlruns/` on your machine.

### Without Docker

Needs Python 3.10+ and Java 17 or 21.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt

python src/generate_claims.py
PYTHONPATH=src python -m pipeline.run_pipeline
PYTHONPATH=src python -m model.train
pytest
mlflow ui --backend-store-uri sqlite:///mlflow.db
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
  model/
    train.py             step 3: train, evaluate, log to MLflow, register
tests/                   27 tests: pipeline rules, end to end, and the model
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
