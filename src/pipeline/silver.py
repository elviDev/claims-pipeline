"""
SILVER layer: clean, typed, trustworthy data.

What happens here:
  1. Convert text columns to proper types (dates, numbers).
  2. Remove duplicate rows.
  3. Fix problems that have a safe, obvious fix (missing channel -> "unknown").
  4. Check every row against the quality rules.
  5. Good rows go to silver. Bad rows go to quarantine, with the reason.

The key decision is FIX vs QUARANTINE:
  * A missing channel doesn't change what the claim is worth, so we fill it in.
  * A missing or negative amount does. Guessing it would be inventing money,
    so that row is quarantined for a human to check.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pipeline.quality import Rule, apply_rules, split_valid_invalid


def safe_cast(column: str, to_type: str):
    """
    Convert text to a type. If the text is invalid ("abc" as a number,
    "2023-13-45" as a date), return NULL instead of crashing the job.

    Spark 4 and Databricks run in ANSI mode, where a plain .cast() on bad
    input throws an error and stops the whole pipeline. One bad row should
    never kill the run: it should become NULL and fail a quality rule.
    """
    return F.expr(f"try_cast(trim({column}) AS {to_type})").alias(column)

# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

POLICY_RULES = [
    Rule("policy_id_present", "policy_id IS NOT NULL", "Every policy needs an ID"),
    Rule("start_date_valid", "start_date IS NOT NULL", "Start date must be a real date"),
    Rule("end_after_start", "end_date > start_date", "Policy must end after it starts"),
    Rule("premium_positive", "annual_premium > 0", "Premium must be above zero"),
]


def clean_policies(bronze: DataFrame) -> DataFrame:
    return (
        bronze
        .select(
            F.trim("policy_id").alias("policy_id"),
            F.lower(F.trim("product")).alias("product"),
            F.trim("region").alias("region"),
            safe_cast("customer_age", "INT"),
            safe_cast("start_date", "DATE"),
            safe_cast("end_date", "DATE"),
            safe_cast("annual_premium", "DOUBLE"),
        )
        .dropDuplicates(["policy_id"])
    )


def build_silver_policies(bronze: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Returns (valid, quarantine, checked). `checked` is used for the quality report."""
    checked = apply_rules(clean_policies(bronze), POLICY_RULES)
    valid, quarantine = split_valid_invalid(checked)
    return valid, quarantine, checked


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

CLAIM_RULES = [
    Rule("claim_amount_present", "claim_amount IS NOT NULL", "Amount is missing"),
    # "IS NULL OR": a missing amount is already caught by the rule above.
    # Without this, one missing value would be counted as two problems.
    Rule("claim_amount_positive", "claim_amount IS NULL OR claim_amount > 0", "Amount must be above zero"),
    Rule("incident_date_valid", "incident_date IS NOT NULL", "Incident date must be a real date"),
    Rule("reported_after_incident", "report_date >= incident_date",
         "A claim can't be reported before the incident happened"),
    Rule("policy_exists", "_policy_found", "Claim must belong to a known policy"),
]


def clean_claims(bronze: DataFrame) -> DataFrame:
    return (
        bronze
        .select(
            F.trim("claim_id").alias("claim_id"),
            F.trim("policy_id").alias("policy_id"),
            safe_cast("incident_date", "DATE"),
            safe_cast("report_date", "DATE"),
            safe_cast("claim_amount", "DOUBLE"),
            # Safe fix: an unknown channel is still useful information
            F.coalesce(F.lower(F.trim("channel")), F.lit("unknown")).alias("channel"),
            F.trim("description").alias("description"),
            safe_cast("is_fraud", "INT"),
        )
        # The same claim exported twice is still one claim
        .dropDuplicates(["claim_id"])
    )


def build_silver_claims(bronze: DataFrame, valid_policies: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Returns (valid, quarantine, checked)."""
    known_policies = valid_policies.select("policy_id").withColumn("_policy_found", F.lit(True))

    claims = (
        clean_claims(bronze)
        # Left join: keep every claim, and mark whether its policy was found
        .join(known_policies, on="policy_id", how="left")
        .withColumn("_policy_found", F.coalesce("_policy_found", F.lit(False)))
    )

    checked = apply_rules(claims, CLAIM_RULES)
    valid, quarantine = split_valid_invalid(checked)
    return valid.drop("_policy_found"), quarantine.drop("_policy_found"), checked
