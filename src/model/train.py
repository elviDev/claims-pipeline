"""
Step 3: train a fraud-risk model on the gold fraud_features table.

What happens here:
  1. Load data/gold/fraud_features (one row per claim) with pandas.
  2. Split into train and test, keeping the same fraud rate in both.
  3. Train two models: a simple baseline and a tree model.
  4. Score both on the test set with metrics that make sense for rare fraud.
  5. Record everything in MLflow and register the better model.

The model is deliberately simple. For a data engineering role the interesting
part is everything around it: a repeatable training run, every result recorded,
and a registered model that the API in step 4 can load by name.

Run it:
    python -m model.train
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw charts to files, no screen needed (Docker has none)

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

# Categories the model sees as "one of these values"
CATEGORICAL = ["product", "region", "channel"]

# Numbers the model sees as amounts
NUMERIC = [
    "customer_age",
    "claim_amount",
    "annual_premium",
    "days_since_policy_start",
    "report_delay_days",
    "amount_vs_product_median",
    "amount_vs_premium",
]

FEATURES = CATEGORICAL + NUMERIC
TARGET = "is_fraud"
# claim_id is left out on purpose: it's a label for a row, not a signal.
# A model could memorise IDs and look great on paper while learning nothing.


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_features(path: str | Path) -> pd.DataFrame:
    """Read the gold table. Spark wrote a folder of parquet files, pandas reads the whole folder."""
    df = pd.read_parquet(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        # Fail loudly here, not with a confusing error deep inside scikit-learn
        raise ValueError(f"fraud_features is missing columns: {sorted(missing)}")

    # Treat every number as a decimal, even ages and day counts. The model's
    # input contract (the MLflow signature) is built from these types, and a
    # whole-number column would make the API reject 40.0 or a missing value.
    df[NUMERIC] = df[NUMERIC].astype("float64")
    return df


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Hold back part of the data to test on. Returns X_train, X_test, y_train, y_test.

    stratify=y keeps the fraud rate the same in both parts. With only ~3% fraud,
    a plain random split could by bad luck put too few fraud cases in the test
    set, and the test scores would be noise.
    """
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# The settings for each model live here as plain dicts, so the exact same
# values can be passed to the model AND logged to MLflow. What's recorded is
# guaranteed to be what was used.
MODEL_PARAMS = {
    # Baseline: a straight-line model. Fast, easy to explain, and it tells us
    # what score a "fancy" model has to beat to be worth its complexity.
    "logistic_regression": {
        "C": 1.0,                      # how strongly to keep weights small (lower = simpler model)
        "max_iter": 1000,
        "class_weight": "balanced",
    },
    # Tree model: builds many small decision trees, each fixing the mistakes of
    # the ones before. It can pick up rules like "early claim AND big amount"
    # that a straight line can't express.
    "hist_gradient_boosting": {
        "learning_rate": 0.05,
        "max_iter": 300,               # number of trees
        "max_leaf_nodes": 15,          # small trees, less chance of memorising noise
        "class_weight": "balanced",
        "random_state": 42,
    },
}
# class_weight="balanced" is how we handle the imbalance. With 3% fraud, the
# model could get 97% of rows right by never flagging anything. "balanced"
# makes each fraud case count ~30x as much in training, so missing fraud is
# expensive and the model has to actually learn what it looks like.


def build_model(name: str) -> Pipeline:
    """
    Data prep + model in one object.

    Saving them together means the API in step 4 can send in a raw claim
    (product="auto", claim_amount=1800, ...) and the model does its own
    encoding. No chance of preparing the data one way in training and
    another way in production.
    """
    params = MODEL_PARAMS[name]

    if name == "logistic_regression":
        # A linear model compares weights across columns, so numbers need a
        # common scale (claim_amount in thousands vs report_delay_days in units).
        numeric_prep = StandardScaler()
        classifier = LogisticRegression(**params)
    elif name == "hist_gradient_boosting":
        # Trees only ask "is this value above or below X?", so scale doesn't matter.
        numeric_prep = "passthrough"
        classifier = HistGradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {name}")

    prep = ColumnTransformer([
        # One column per category value: product=auto -> product_auto=1.
        # handle_unknown="ignore": a new region showing up in production
        # becomes all zeros instead of crashing the API.
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("numbers", numeric_prep, NUMERIC),
    ])

    return Pipeline([("prep", prep), ("classifier", classifier)])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Investigators can check about 1 in 20 claims. The model's job is to choose
# which ones. This is a business decision, so it's a setting, not a constant
# buried in the maths.
DEFAULT_REVIEW_RATE = 0.05


def fraud_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Fraud score per claim, between 0 and 1. Higher means more suspicious.

    Because of class_weight="balanced" these scores are pushed upwards: they
    rank claims well, but 0.3 does NOT mean "30% chance of fraud". That's
    fine here, because we only use them to rank and to compare with a threshold.
    """
    return model.predict_proba(X)[:, 1]


def pick_threshold(model: Pipeline, X_train: pd.DataFrame, review_rate: float) -> float:
    """
    The score above which a claim gets flagged, set so that about `review_rate`
    of claims are flagged.

    Worked out on the TRAINING data. Choosing it by looking at the test set
    would be tuning on the exam, and the test scores would come out too good.
    """
    return float(np.quantile(fraud_scores(model, X_train), 1 - review_rate))


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float) -> tuple[dict, np.ndarray]:
    """
    Score the model on held-out data. Returns (metrics, confusion matrix).

    Why not just accuracy? With ~3% fraud, a "model" that says "not fraud"
    to every claim is 97% accurate and catches nothing. We still record
    accuracy next to that do-nothing number, precisely to show how useless it is.

    What we use instead:
      roc_auc   - pick one fraud and one honest claim at random: how often does
                  the fraud get the higher score? 0.5 = coin flip, 1.0 = perfect.
      pr_auc    - average precision. Focuses only on how well fraud rises to
                  the top of the list. A random model scores the fraud rate
                  (~0.03), so that's the number to beat. The main metric here,
                  because with rare fraud the top of the list is all that matters.
      precision - of the claims we flagged, how many were really fraud?
                  (low = investigators waste time on honest customers)
      recall    - of all the fraud, how much did we flag?
                  (low = fraud gets paid out)
    """
    scores = fraud_scores(model, X)
    flagged = (scores >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y, scores),
        "pr_auc": average_precision_score(y, scores),
        "precision": precision_score(y, flagged, zero_division=0),
        "recall": recall_score(y, flagged),
        "flagged_rate": flagged.mean(),
        "accuracy": accuracy_score(y, flagged),
        "accuracy_if_never_fraud": 1 - y.mean(),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}

    # Rows = what really happened, columns = what the model said:
    #   [[honest, not flagged   honest, flagged     ]
    #    [fraud,  not flagged   fraud,  flagged     ]]
    matrix = confusion_matrix(y, flagged, labels=[0, 1])
    return metrics, matrix


def plot_confusion_matrix(matrix: np.ndarray, title: str):
    """A picture of the confusion matrix, to save with the run in MLflow."""
    display = ConfusionMatrixDisplay(matrix, display_labels=["honest", "fraud"])
    fig, ax = plt.subplots(figsize=(4, 4))
    display.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(title)
    ax.set_xlabel("Model said")
    ax.set_ylabel("Really was")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# MLflow: record every run, register the winner
# ---------------------------------------------------------------------------

EXPERIMENT = "claims-fraud"
REGISTERED_MODEL = "claims-fraud-model"
CHAMPION_ALIAS = "champion"

# Run records (settings, metrics, registry) go in a small SQLite database file.
# The model files themselves go in mlruns/. Both are git-ignored: they're
# outputs, and anyone can recreate them by running this script.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

# MLflow saves the model with skops, not pickle. Loading a pickle file can run
# any code hidden inside it, so a tampered model file could take over the API
# server. skops only rebuilds object types on an allow-list. The tree model
# uses one type that isn't on the default list; we trained it ourselves, so we
# add it here, and nothing else.
SKOPS_TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def train_and_log(name: str, data: pd.DataFrame, features_path: str, review_rate: float) -> tuple[str, dict]:
    """
    Train one model inside an MLflow run. Returns (model URI, test metrics).

    Everything needed to answer "where did this model come from and how good
    is it?" is saved with the run: settings, data, scores, and the model itself.
    """
    X_train, X_test, y_train, y_test = split(data)

    with mlflow.start_run(run_name=name):
        # What went in: the model settings plus how we set up the experiment
        mlflow.log_params({"model_type": name, **MODEL_PARAMS[name]})
        mlflow.log_params({
            "review_rate": review_rate,
            "features": ",".join(FEATURES),
            "n_train": len(X_train),
            "n_test": len(X_test),
        })
        # Which data it learned from. MLflow stores a fingerprint of the table,
        # so later you can tell if two models were trained on the same data.
        # (MLflow warns here about the whole-number is_fraud column and the
        # path format. Neither affects the model, so we hide just these warnings.)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dataset = mlflow.data.from_pandas(data, source=features_path, name="fraud_features", targets=TARGET)
            mlflow.log_input(dataset, context="training")

        model = build_model(name).fit(X_train, y_train)
        threshold = pick_threshold(model, X_train, review_rate)
        metrics, matrix = evaluate(model, X_test, y_test, threshold)
        metrics["threshold"] = round(threshold, 4)

        # What came out
        mlflow.log_metrics(metrics)
        fig = plot_confusion_matrix(matrix, name)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # The model itself, with a signature: the column names and types it
        # expects in, and what it returns. The API in step 4 can check incoming
        # requests against it. The input example is a few real rows, stored so
        # anyone can see what a valid request looks like.
        # pyfunc_predict_fn="predict_proba": when loaded as a generic MLflow
        # model, return fraud scores rather than a hard yes/no.
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_train, model.predict_proba(X_train.head(5))),
            input_example=X_train.head(3),
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    return model_info.model_uri, metrics


def register_best(results: dict[str, tuple[str, dict]], review_rate: float) -> tuple[str, str]:
    """
    Register the model with the best PR AUC and point the "champion" alias at it.

    The API in step 4 loads "models:/claims-fraud-model@champion". To put a new
    model live, you move the alias; the API code never changes. To roll back,
    move it back to the previous version.
    """
    best_name = max(results, key=lambda name: results[name][1]["pr_auc"])
    model_uri, metrics = results[best_name]

    version = mlflow.register_model(model_uri, REGISTERED_MODEL)

    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION_ALIAS, version.version)
    # The flag/don't-flag threshold belongs with the model: the API needs both.
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "threshold", str(metrics["threshold"]))
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "review_rate", str(review_rate))
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "model_type", best_name)

    return best_name, version.version


def run(
    features_path: str | Path = "data/gold/fraud_features",
    review_rate: float = DEFAULT_REVIEW_RATE,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> dict:
    """Train every model, log each run, register the best. Returns the test metrics per model."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    data = load_features(features_path)
    print(f"Loaded {len(data):,} claims, fraud rate {data[TARGET].mean():.1%}")

    results = {}
    for name in MODEL_PARAMS:
        print(f"Training {name}...")
        results[name] = train_and_log(name, data, str(features_path), review_rate)

    best_name, version = register_best(results, review_rate)

    print(f"\nTest set results (flagging the riskiest {review_rate:.0%} of claims):")
    for name, (_, metrics) in results.items():
        print(f"  {name:24s} " + "  ".join(f"{k}={metrics[k]:.3f}" for k in ["roc_auc", "pr_auc", "precision", "recall"]))
    print(f"\nRegistered {best_name} as {REGISTERED_MODEL} version {version} (@{CHAMPION_ALIAS})")

    return {name: metrics for name, (_, metrics) in results.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud model and log it to MLflow.")
    parser.add_argument("--features-path", default="data/gold/fraud_features")
    parser.add_argument("--review-rate", type=float, default=DEFAULT_REVIEW_RATE,
                        help="Share of claims investigators can review (sets the threshold)")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()

    run(args.features_path, args.review_rate, args.tracking_uri)


if __name__ == "__main__":
    main()
