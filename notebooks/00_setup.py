# Databricks notebook source
# MAGIC %md
# MAGIC # 00 Setup: schema and volume
# MAGIC
# MAGIC Run once. Creates the places our data lives in **Unity Catalog**, which names
# MAGIC everything in three levels: `catalog.schema.object`.
# MAGIC
# MAGIC | What | Name | Holds |
# MAGIC |------|------|-------|
# MAGIC | Schema | `workspace.claims` | all our tables (bronze, silver, quarantine, gold) and the model |
# MAGIC | Volume | `workspace.claims.raw` | the raw CSV files, at `/Volumes/workspace/claims/raw/` |
# MAGIC
# MAGIC Tables hold rows; a **volume** holds files. Raw CSVs are files, so they go in a volume.
# MAGIC `IF NOT EXISTS` makes this safe to run again.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "claims")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema} "
          "COMMENT 'Insurance claims pipeline: medallion tables and the fraud model'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw "
          "COMMENT 'Raw CSV files as they arrive from the source systems'")

display(spark.sql(f"SHOW VOLUMES IN {catalog}.{schema}"))
