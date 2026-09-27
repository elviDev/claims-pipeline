# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Generate: synthetic raw data into the volume
# MAGIC
# MAGIC Runs the same generator as locally (`src/generate_claims.py`), with the same
# MAGIC injected data problems, and writes `policies.csv` and `claims.csv` into the
# MAGIC volume. In a real system this is where files would land from the policy and
# MAGIC claims systems.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "claims")
dbutils.widgets.text("n_policies", "5000")
dbutils.widgets.text("n_claims", "20000")
dbutils.widgets.text("seed", "42")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# Make src/ importable. The notebook runs from <repo>/notebooks, so the repo
# root is one folder up. Works in the Git folder and when deployed by a bundle.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "src"))

# COMMAND ----------

from pathlib import Path

from generate_claims import main

raw_dir = f"/Volumes/{catalog}/{schema}/raw"
main(
    n_policies=int(dbutils.widgets.get("n_policies")),
    n_claims=int(dbutils.widgets.get("n_claims")),
    out_dir=Path(raw_dir),
    seed=int(dbutils.widgets.get("seed")),
)

# COMMAND ----------

display(dbutils.fs.ls(raw_dir))
