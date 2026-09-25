"""
GOLD layer: tables built for a specific use.

Silver is clean data in its original shape. Gold reshapes it for whoever uses it:

  * claims_monthly  -> for business reporting (a dashboard, a monthly report)
  * fraud_features  -> for the fraud model in step 3, one row per claim, only
                       numbers and categories a model can learn from
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def build_claims_monthly(claims: DataFrame, policies: DataFrame) -> DataFrame:
    """Claims count, cost and fraud rate by month, product and region."""
    return (
        claims
        .join(policies.select("policy_id", "product", "region"), on="policy_id")
        .withColumn("month", F.date_trunc("month", "incident_date").cast("date"))
        .groupBy("month", "product", "region")
        .agg(
            F.count("*").alias("n_claims"),
            F.round(F.sum("claim_amount"), 2).alias("total_amount"),
            F.round(F.avg("claim_amount"), 2).alias("avg_amount"),
            F.round(F.avg("is_fraud"), 4).alias("fraud_rate"),
        )
        .orderBy("month", "product", "region")
    )


def build_fraud_features(claims: DataFrame, policies: DataFrame) -> DataFrame:
    """
    One row per claim with the signals an investigator would look at.

    Each feature is a question a fraud analyst would ask:
      days_since_policy_start  -> did they claim right after buying the policy?
      report_delay_days        -> why did it take so long to report?
      amount_vs_product_median -> is this much bigger than normal for this product?
      amount_vs_premium        -> are they claiming many times what they pay?
    """
    by_product = Window.partitionBy("product")

    return (
        claims
        .join(policies, on="policy_id")
        .withColumn("days_since_policy_start", F.datediff("incident_date", "start_date"))
        .withColumn("report_delay_days", F.datediff("report_date", "incident_date"))
        .withColumn(
            "amount_vs_product_median",
            F.round(F.col("claim_amount") / F.median("claim_amount").over(by_product), 3),
        )
        .withColumn("amount_vs_premium", F.round(F.col("claim_amount") / F.col("annual_premium"), 3))
        .select(
            "claim_id",
            "product",
            "region",
            "channel",
            "customer_age",
            "claim_amount",
            "annual_premium",
            "days_since_policy_start",
            "report_delay_days",
            "amount_vs_product_median",
            "amount_vs_premium",
            "is_fraud",
        )
    )
