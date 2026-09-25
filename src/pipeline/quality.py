"""
Data quality rules.

A rule is a name plus a condition that must be TRUE for a good row.
The condition is written in SQL, e.g. "claim_amount > 0". That keeps the rules
easy to read, and means they could later live in a config file instead of code.

Rows that break any rule are not deleted. They go to a quarantine table
together with the list of rules they failed, so someone can look at them,
fix the source, and reprocess.

Deleting bad rows silently is the classic mistake: the numbers downstream
look fine, but nobody knows 3% of the claims just vanished.
"""

from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass(frozen=True)
class Rule:
    name: str
    condition: str             # SQL expression. TRUE means the row passes
    description: str = ""


def apply_rules(df: DataFrame, rules: list[Rule]) -> DataFrame:
    """Add a `_failed_rules` column: an array with the name of every rule the row broke."""
    checks = [
        # coalesce(..., False): if the condition is NULL (e.g. comparing a missing
        # date), count it as a failure, not as a pass.
        F.when(~F.coalesce(F.expr(rule.condition), F.lit(False)), F.lit(rule.name))
        for rule in rules
    ]
    failed = F.filter(F.array(*checks), lambda x: x.isNotNull())
    return df.withColumn("_failed_rules", failed)


def split_valid_invalid(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Split a DataFrame that has `_failed_rules` into (valid, quarantine)."""
    valid = df.filter(F.size("_failed_rules") == 0).drop("_failed_rules")
    quarantine = df.filter(F.size("_failed_rules") > 0)
    return valid, quarantine


def quality_report(checked: DataFrame, rules: list[Rule]) -> dict:
    """Count how many rows failed each rule. Returns a plain dict (easy to log or save as JSON)."""
    exploded = checked.select(F.explode("_failed_rules").alias("rule"))
    counts = {row["rule"]: row["count"] for row in exploded.groupBy("rule").count().collect()}

    total = checked.count()
    invalid = checked.filter(F.size("_failed_rules") > 0).count()

    return {
        "total_rows": total,
        "valid_rows": total - invalid,
        "quarantined_rows": invalid,
        "quarantine_rate": round(invalid / total, 4) if total else 0.0,
        "failures_by_rule": {rule.name: counts.get(rule.name, 0) for rule in rules},
    }
