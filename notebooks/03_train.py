# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Train: fraud model, tracked in MLflow, registered in Unity Catalog
# MAGIC
# MAGIC The same training code as locally (`src/model/train.py`). What changes:
# MAGIC
# MAGIC | | Local | Databricks |
# MAGIC |---|---|---|
# MAGIC | Features from | `data/gold/fraud_features` (parquet) | table `workspace.claims.gold_fraud_features` |
# MAGIC | Runs tracked in | `mlflow.db` (SQLite) | the workspace (`tracking_uri="databricks"`) |
# MAGIC | Experiment | `claims-fraud` | `/Users/<you>/claims-fraud` (experiments live in the workspace like files) |
# MAGIC | Model registry | local SQLite | **Unity Catalog** (`registry_uri="databricks-uc"`) |
# MAGIC | Model name | `claims-fraud-model` | `workspace.claims.claims_fraud_model` |
# MAGIC
# MAGIC In Unity Catalog the model sits next to the tables it was trained on, with the
# MAGIC same three-level name and the same permissions. The winner gets the `champion`
# MAGIC alias, exactly as locally.

# COMMAND ----------

# The same library versions as our Docker image (requirements.txt), so a model
# trained here behaves the same as one trained locally, and our API could load it.
# pyspark is NOT installed: Databricks has its own.
%pip install mlflow==3.16.1 scikit-learn==1.9.1 skops==0.16.0

# COMMAND ----------

# Restart Python so the newly installed versions are the ones that get imported
dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "claims")
dbutils.widgets.text("review_rate", "0.05")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# Make src/ importable (the notebook runs from <repo>/notebooks)
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "src"))

# COMMAND ----------

from model.train import run

features_table = f"{catalog}.{schema}.gold_fraud_features"
user = spark.sql("SELECT current_user()").first()[0]

# ~20,000 rows: small enough for pandas and scikit-learn on one machine.
# Spark did the heavy lifting (cleaning, joining, features); the model doesn't need it.
features = spark.table(features_table).toPandas()

metrics = run(
    features,
    review_rate=float(dbutils.widgets.get("review_rate")),
    tracking_uri="databricks",
    experiment=f"/Users/{user}/claims-fraud",
    registry_uri="databricks-uc",
    model_name=f"{catalog}.{schema}.claims_fraud_model",
    source=features_table,
)

# COMMAND ----------

import pandas as pd

display(pd.DataFrame(metrics).T.reset_index(names="model"))
