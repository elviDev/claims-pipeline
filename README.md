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
| 2 | Bronze / silver / gold layers in PySpark, with data quality checks | ⏳ |
| 3 | Fraud-risk model, tracked with MLflow | ⏳ |
| 4 | FastAPI service that scores a claim, running in Docker | ⏳ |
| 5 | LLM step: extract damage type and urgency from the claim description | ⏳ |
| 6 | GitHub Actions: tests and Docker build on every push | ⏳ |
| 7 | Run the pipeline on Databricks | ⏳ |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/generate_claims.py    # writes data/raw/policies.csv and claims.csv
pytest -q                        # run the tests
```

## The data

**policies.csv**: one row per policy (product, region, customer age, start and end date, premium).

**claims.csv**: one row per claim (policy, incident and report dates, amount,
channel, free-text description, fraud label).

The fraud label follows real red flags: claims filed in the first month of a
policy, amounts far above normal for the product, and incidents reported late.

The raw data is intentionally dirty, the way real exports are:

- missing amounts and channels
- negative amounts
- claims reported before the incident happened
- claims pointing to policies that don't exist
- duplicate rows

Step 2 is about catching and handling all of these.
