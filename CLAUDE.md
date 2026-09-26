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
| 4    | FastAPI service that scores a claim, in Docker (`src/api/`)                                | done     |
| 5    | LLM extracts damage type and urgency from the claim description (`src/llm/`)               | done     |
| 6    | GitHub Actions: tests and Docker build on every push                                       | **next** |
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

## Step 4 as built (notes for step 5)

- `src/api/main.py`: `Scorer` holds model + version + threshold + medians, loaded
  once in the lifespan by resolving the alias to a version first. Endpoints
  `/health`, `/score`, `/score/batch`. Add `/claims/triage` next to them.
- `src/api/features.py`: `build_features(raw_df, medians)` mirrors gold.py.
  `tests/test_features.py` has the Spark-vs-API skew test.
- `train.py` logs `product_medians.json` in each run (`MEDIANS_ARTIFACT`).
- Shared test fixtures in `tests/conftest.py`: `features_path` and
  `trained_registry` (session scoped, temp MLflow registry). `test_api.py`
  starts the app with `TestClient` against it.
- Starlette wants `httpx2` (not `httpx`) for `TestClient`.
- MLflow 3 returns model version numbers as int; the API reports them as str.
- `docker compose up api` on port 8000, healthcheck calls `/health`.
- Docker Desktop has been slow at times; long test runs can exceed 10 minutes
  to start. Run them in the background.

## Step 4 original plan: FastAPI scoring service in Docker

Goal: a claims system sends a new claim, the API answers with a fraud score and
whether it should go to a human for review. The Swagger page (`/docs`) is the
live demo screen at the job fair.

Elvis already knows FastAPI well (see his ai-document-intelligence repo), so
spend the explaining on the MLOps parts, not the FastAPI basics.

Build:

- `src/api/main.py`
  - Load the model **once at startup** (FastAPI lifespan), from
    `models:/claims-fraud-model@champion`. Read the `threshold` and model version
    from the registry with `MlflowClient`. Model URI and tracking URI come from
    env vars (`MODEL_URI`, `MLFLOW_TRACKING_URI`) with the current defaults.
  - `GET /health`: status plus which model version is loaded. This is what a
    load balancer or Kubernetes probe would call.
  - `POST /score`: one claim in, returns `fraud_score`, `flag_for_review`
    (score >= threshold), `threshold`, `model_version`.
  - `POST /score/batch`: a list of claims, for a nightly job.
- **The request is a raw claim, not the model features.** The claims system
  knows dates, amounts and the policy; it doesn't know `days_since_policy_start`
  or `amount_vs_product_median`. So the API computes the features itself in
  `src/api/features.py`.
  - This is the main lesson of the step: **training/serving skew**. If the API
    computes a feature slightly differently from the gold layer, the model gets
    inputs it never saw in training and the scores are quietly wrong, with no error.
  - `amount_vs_product_median` needs the per-product medians from training.
    Small change to `train.py`: save the medians as a JSON artifact with the
    model (e.g. `product_medians.json`), and the API loads them together with
    the model. Then the medians always match the model version.
  - Add a **skew test**: take some claims, compute features with the Spark gold
    code and with the API code, assert they match.
- Request model (pydantic): `product`, `region`, `channel` as `Literal[...]` of
  the known values; dates as `date`; amounts as `float` with `gt=0`; reject a
  report date before the incident date. The same rules as the silver layer, so
  bad input gets a clear 422 instead of a meaningless score.
- Log one line per prediction (claim_id, score, flagged, model version, latency).
  That log is the raw material for monitoring later (MLOps Zoomcamp module 5).
- Docker: add an `api` service to `docker-compose.yml` using the same image,
  port 8000, command `uvicorn api.main:app --host 0.0.0.0 --port 8000`. Add
  `fastapi`, `uvicorn[standard]`, `httpx` to requirements.
- `tests/test_api.py` with `TestClient`: train a tiny model into a temporary
  MLflow tracking URI in a fixture (reuse the helpers from `test_model.py`),
  then test: health works, a normal claim gets a score between 0 and 1, a claim
  with red flags scores higher than a normal one, bad input returns 422.
- README: "Scoring API" section with a curl example and a screenshot of `/docs`.

Worth explaining:

- Why load the model at startup and not per request (speed; one fixed version).
- Why the API loads `@champion` and not a version number: promoting or rolling
  back a model is moving the alias, with no code change or redeploy.
- Training/serving skew, and how the skew test and shared medians prevent it.
- What would change in production: an MLflow server or Databricks Model Serving
  instead of a local sqlite file, the model baked into the image or pulled at
  deploy, several API replicas behind a load balancer (Kubernetes), health
  checks.

## Step 5 as built (notes for step 6)

- LLM: Ollama on Elvis's Windows host (CPU only), reached from Docker at
  `http://host.docker.internal:11434/v1`. Models: qwen2.5:3b (best), gemma3:1b,
  qwen2.5:0.5b. Settings in `.env` (git- and docker-ignored), names in
  `.env.example`. Compose loads it with `env_file` `required: false`.
- `src/llm/extract.py`: `FakeExtractor` (keywords from the generator templates
  only), `LLMExtractor` (openai package, temperature 0, json_schema response
  format, pydantic validation, one retry that includes the error, fallback with
  `needs_manual_review`, sha256 cache, openai `max_retries=0`), `get_extractor()`
  picks the LLM only when `LLM_API_KEY` and `LLM_MODEL` are set.
- `src/llm/evaluate.py`: 29 cases (first 18 = templates, checked by a test),
  MLflow experiment `claims-llm-extraction`, `PROMPT_VERSION = "v1"`, untimed
  warm-up call. Results table is in the README.
- CI must not call an LLM: without `.env` there is no `LLM_API_KEY`, so the fake
  extractor is used. `tests/test_api.py` also removes `LLM_API_KEY`.
- Don't tune the prompt or keywords on the 29 eval cases; a v2 prompt needs a
  separate held-out set.

## Step 5 original plan: LLM reads the claim description

Goal: turn the free-text description ("Burst pipe in the bathroom flooded the
hallway.") into structured fields a claims handler can route on. This covers
the "LLM products" part of the job. Elvis knows the OpenAI API and RAG already.

Build:

- `src/llm/schema.py`: pydantic model for the output:
  `damage_type` (Literal: collision, theft, water, fire, weather, lost_luggage,
  travel_disruption, medical, other), `urgency` (low / medium / high),
  `injury_mentioned` (bool), `third_party_involved` (bool), `summary` (short str).
- `src/llm/extract.py`
  - A small interface (`Extractor` with `extract(description) -> ClaimInfo`)
    with two implementations: a real LLM one, and a `FakeExtractor` (simple
    keyword rules) used in tests and when no API key is set. Tests and CI must
    never call a paid API.
  - Ask Elvis which provider he has a key for; keep the provider code in one
    place so it's easy to swap.
  - Use the provider's structured output / JSON mode, then **validate with
    pydantic anyway**. If the output is invalid, retry once, then return
    `damage_type="other"` with a `needs_manual_review` flag. Never pass
    unvalidated LLM output downstream.
  - Temperature 0, so the same description gives the same answer.
  - Cache by a hash of the description. The generator reuses templates, and in
    real life cost and latency matter.
  - API key from `.env` (already git-ignored), loaded via `env_file` in compose.
    Add a `.env.example` with the variable name and no value.
- **Evaluation** (`src/llm/evaluate.py`): the generator's description templates
  are known, so build a labelled set (template -> expected damage_type and
  urgency, ~20 to 30 cases, include a few tricky ones). Run the extractor,
  measure accuracy per field, log the results to MLflow as its own experiment
  (`claims-llm-extraction`) with the prompt version as a param. "How do you know
  the LLM works? I measured it" is the point to make in the interview.
- Wire it in:
  - API: `POST /claims/triage` returns the fraud score and the extracted fields
    in one answer. If no LLM key is configured, use the fake extractor and say
    so in the response.
  - Optional batch: enrich silver claims into a `gold/claims_enriched` table.
- Tests with the fake extractor and a mocked client: valid output parses,
  invalid JSON triggers the fallback, the cache avoids a second call.
- README: "LLM extraction" section with the eval results table.

Worth explaining:

- Why structured output plus validation (LLMs sometimes return junk).
- Why a fake extractor (tests must be fast, free and work offline).
- Why evaluate with a labelled set, and track it in MLflow like the fraud model.
- **Privacy**, important for an insurer: claim descriptions contain personal
  data. In a real deployment you'd mask names and addresses before sending
  text out, or use a model hosted inside the company's cloud in the EU (for
  example Azure OpenAI or AWS Bedrock in an EU region). GDPR applies.
- RAG vs this: this is extraction, not retrieval. RAG would fit a different
  feature, like answering "is this covered by my policy?" from policy documents,
  which ties back to his existing RAG project.
