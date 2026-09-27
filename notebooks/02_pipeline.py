# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Pipeline: bronze, silver, quarantine, gold as Unity Catalog tables
# MAGIC
# MAGIC The same pipeline code as locally (`src/pipeline/`), in **table mode**: each
# MAGIC medallion layer becomes a managed Delta table in `workspace.claims`.
# MAGIC
# MAGIC | Layer | Tables |
# MAGIC |-------|--------|
# MAGIC | Bronze: raw data as received, plus load time and source file | `bronze_policies`, `bronze_claims` |
# MAGIC | Silver: typed, deduplicated, passed every quality rule | `silver_policies`, `silver_claims` |
# MAGIC | Quarantine: failed a rule, with the list of rules it failed | `quarantine_policies`, `quarantine_claims` |
# MAGIC | Gold: built for one use | `gold_claims_monthly` (reporting), `gold_fraud_features` (the model) |
# MAGIC
# MAGIC **Quality gate:** if more than 5% of claims fail the rules, this notebook fails
# MAGIC before writing gold. In the Job, a failed task stops the tasks after it, so
# MAGIC the model never trains on a broken load.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "claims")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# Make src/ importable (the notebook runs from <repo>/notebooks)
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "src"))

# COMMAND ----------

import json

from pipeline.run_pipeline import run
from pipeline.storage import Storage

storage = Storage("table", f"{catalog}.{schema}")
# `spark` is the session Databricks already gives every notebook
report = run(storage, raw_dir=f"/Volumes/{catalog}/{schema}/raw", spark=spark)

print(json.dumps(report, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## What landed where

# COMMAND ----------

display(spark.sql(f"SHOW TABLES IN {catalog}.{schema}"))

# COMMAND ----------

display(spark.table(storage.location("gold", "fraud_features")).limit(10))
