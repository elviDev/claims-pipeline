# Project context for Claude

## Why this project exists

Elvis is building this as a portfolio project for a **Data Engineer (MLOps / LLM
products)** role at **ServiZurich**, the Zurich Insurance Group tech centre in
Barcelona. He'll demo it at the 22@Network job fair on **21 October 2026**, in
short 5 to 15 minute conversations. The job asks for: Spark / Databricks, data
quality, Docker / Kubernetes, CI/CD, MLOps (model deployment, MLflow), LLM
products, and AWS.

Elvis's background: Python, FastAPI, PostgreSQL, pgvector, RAG, Docker, pytest.
Previously an Application Engineer at a bank. Newer to Spark and MLOps.

## How to work with Elvis

- Build **step by step**. Each step should be small enough for him to read and
  understand fully, because interviewers will ask him about every piece.
- Explain the _why_ behind every design choice, in plain language, with code
  comments that read like a colleague explaining it.
- Test everything before handing it over. Run pytest after each change.
- He commits and pushes after each step himself.
- Writing style: plain and direct, like explaining to a smart friend. No buzzwords,
  no corporate jargon, no em dashes.

## His setup

- Windows, VS Code, **Git Bash** terminal (use forward slashes, and
  `source .venv/Scripts/activate` to activate the venv).
- Docker Desktop. Spark runs through `docker compose` because Spark on Windows
  needs extra Hadoop tools (winutils) to write files.

## Project conventions

- Code in `src/`, tests in `tests/`, `pytest.ini` sets `pythonpath = src`.
- Spark 4.0 (ANSI mode on): convert text with `try_cast`, never plain `.cast()`.
- Data quality rules are SQL strings (`Rule(name, "claim_amount > 0", description)`).
- Bad rows go to quarantine with their failed rules, never silently dropped.
- `data/` is git-ignored. Anyone can rebuild it with the generator plus the pipeline.

## Roadmap

| Step | What                                                                                       | Status   |
| ---- | ------------------------------------------------------------------------------------------ | -------- |
| 1    | Synthetic claims and policies data with injected data problems (`src/generate_claims.py`)  | done     |
| 2    | PySpark bronze / silver / gold + quality rules, quarantine, quality gate (`src/pipeline/`) | done     |
| 3    | Fraud-risk model on `data/gold/fraud_features`, tracked with MLflow (`src/model/`)         | done     |
| 4    | FastAPI service that scores a claim, in Docker                                             | **next** |
| 5    | LLM extracts damage type and urgency from the claim description                            |          |
| 6    | GitHub Actions: tests and Docker build on every push                                       |          |
| 7    | Run the pipeline on Databricks Free Edition (Delta tables, `--format delta`)               |          |

## Step 3 as built (notes for step 4)

- Tests run in Docker (`docker compose run --rm pipeline pytest`); the local venv
  is Python 3.14 without pyspark.
- MLflow 3 tracking is `sqlite:///mlflow.db` (git-ignored), model files in
  `mlruns/`. The file-based `mlruns/` store is deprecated. The UI runs in Docker
  (`docker compose up mlflow`) because stored paths are `/app/mlruns/...`.
- Load the model with `models:/claims-fraud-model@champion`. Its threshold is a
  model version tag (`threshold`), plus `review_rate` and `model_type` tags.
- The signature expects all numeric features as double; MLflow rejects int64.
  The FastAPI request model should declare them as `float`.
- Models are saved with skops; `SKOPS_TRUSTED_TYPES` in `train.py` lists the
  extra type the tree model needs.
- Scores come from `predict_proba` with `class_weight="balanced"`, so they rank
  well but are not calibrated probabilities.
- Model scores are close to the ceiling of the generated data (56% of fraud has
  no red flag). Oracle ROC AUC 0.656 / PR AUC 0.089 on the test set.

## Step 3 original plan: fraud model + MLflow

Input: `data/gold/fraud_features` (parquet, one row per claim). Columns:
`claim_id, product, region, channel, customer_age, claim_amount, annual_premium,
days_since_policy_start, report_delay_days, amount_vs_product_median,
amount_vs_premium, is_fraud`. Fraud rate is about 3%.

Build:

- `src/model/train.py`: load features with pandas, stratified train/test split,
  scikit-learn Pipeline (OneHotEncoder for product/region/channel, numeric
  features passed through or scaled), train a LogisticRegression baseline and a
  tree model (e.g. HistGradientBoostingClassifier). Handle the class imbalance
  (class_weight="balanced").
- Metrics that make sense for rare fraud: ROC AUC, PR AUC (average precision),
  precision and recall at the chosen threshold, confusion matrix. Explain why
  accuracy is misleading at 3% fraud (predicting "never fraud" scores 97%).
- MLflow: log params, metrics, the confusion matrix as an artifact, and the model
  with a signature and input example. Register the best model in the local model
  registry as `claims-fraud-model`. Local tracking in `mlruns/` (already
  git-ignored). View with `mlflow ui`.
- `tests/test_model.py`: model trains on a small generated sample, predictions are
  probabilities between 0 and 1, the saved model loads and scores one claim.
- Add `scikit-learn` and `mlflow` to requirements.txt.
- Update README (roadmap, a "Fraud model" section with results and how to run it).

Things worth explaining to Elvis in step 3:

- Why a simple baseline first (so you know if the fancy model is actually better).
- Why PR AUC matters more than accuracy for rare events.
- What MLflow gives you: every run recorded with its settings and results, so
  you can compare runs and know exactly which model is deployed.
- The model is deliberately simple. For a data engineer role, the value is the
  pipeline, tracking and deployment around it, not squeezing out accuracy.
- Honest caveat: `amount_vs_product_median` is computed over all claims before
  the split, a mild form of leakage. Fine for a demo, and a good thing to be
  able to point out if asked.
