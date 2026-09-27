# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Explore: SQL on the gold tables
# MAGIC
# MAGIC Plain SQL on the tables the pipeline built. Anyone with access to the schema
# MAGIC can run these, in a notebook or the SQL editor, without knowing Python.
# MAGIC
# MAGIC Tip: under each result, click **+** next to "Table" and pick **Visualization**
# MAGIC to turn it into a chart.
# MAGIC
# MAGIC The queries use `workspace.claims`. If you changed the catalog or schema, edit them.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fraud rate by product
# MAGIC About 3% of claims are fraud overall. Which products are worse?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   product,
# MAGIC   COUNT(*)                              AS claims,
# MAGIC   SUM(is_fraud)                         AS fraud_cases,
# MAGIC   ROUND(100 * AVG(is_fraud), 2)         AS fraud_rate_pct,
# MAGIC   ROUND(AVG(claim_amount), 0)           AS avg_amount_eur
# MAGIC FROM workspace.claims.gold_fraud_features
# MAGIC GROUP BY product
# MAGIC ORDER BY fraud_rate_pct DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Claims cost per month
# MAGIC From the reporting table. A line chart of `total_cost_eur` by `month` works well.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   month,
# MAGIC   SUM(n_claims)                                              AS claims,
# MAGIC   ROUND(SUM(total_amount), 0)                                AS total_cost_eur,
# MAGIC   ROUND(100 * SUM(fraud_rate * n_claims) / SUM(n_claims), 2) AS fraud_rate_pct
# MAGIC FROM workspace.claims.gold_claims_monthly
# MAGIC GROUP BY month
# MAGIC ORDER BY month

# COMMAND ----------

# MAGIC %md
# MAGIC ## What got quarantined, and why
# MAGIC Every quarantined row keeps the list of rules it broke (`_failed_rules`), so
# MAGIC nothing disappears silently. `explode` turns that list into one row per rule.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT rule, COUNT(*) AS claims
# MAGIC FROM workspace.claims.quarantine_claims
# MAGIC LATERAL VIEW explode(_failed_rules) AS rule
# MAGIC GROUP BY rule
# MAGIC ORDER BY claims DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT claim_id, policy_id, claim_amount, incident_date, report_date, _failed_rules
# MAGIC FROM workspace.claims.quarantine_claims
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta: every write is a version
# MAGIC Each pipeline run overwrites the tables, but Delta keeps the history.
# MAGIC You can query an older version with `VERSION AS OF` ("time travel"), for
# MAGIC example to compare today's silver table with yesterday's.

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY workspace.claims.silver_claims

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Row count now and in the table's first version. Rerunning the pipeline
# MAGIC -- should give the same count every time (overwrite, never append).
# MAGIC SELECT 'current' AS version, COUNT(*) AS claims FROM workspace.claims.silver_claims
# MAGIC UNION ALL
# MAGIC SELECT 'first', COUNT(*) FROM workspace.claims.silver_claims VERSION AS OF 0
