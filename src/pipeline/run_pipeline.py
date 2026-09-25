"""
Run the whole pipeline: raw CSV -> bronze -> silver (+ quarantine) -> gold.

Locally / in Docker:
    python -m pipeline.run_pipeline

On Databricks (step 7) the same code runs, pointed at a Volume and writing Delta:
    python -m pipeline.run_pipeline --base-path /Volumes/main/claims/data --format delta
"""

import argparse
import json
import sys
from pathlib import Path

from pipeline.bronze import read_raw_csv
from pipeline.gold import build_claims_monthly, build_fraud_features
from pipeline.silver import CLAIM_RULES, POLICY_RULES, build_silver_claims, build_silver_policies
from pipeline.quality import quality_report
from pipeline.spark import get_spark

# Quality gate: if more than this share of claims fails the rules, stop.
# Something is badly wrong with the source, and loading it would poison
# every report and model downstream.
MAX_QUARANTINE_RATE = 0.05


def write(df, path: str, fmt: str) -> None:
    df.write.mode("overwrite").format(fmt).save(path)


def run(base_path: str, fmt: str = "parquet") -> dict:
    spark = get_spark()
    base = base_path.rstrip("/")

    # ---- Bronze ----------------------------------------------------------
    print("Bronze: loading raw files...")
    bronze_policies = read_raw_csv(spark, f"{base}/raw/policies.csv")
    bronze_claims = read_raw_csv(spark, f"{base}/raw/claims.csv")
    write(bronze_policies, f"{base}/bronze/policies", fmt)
    write(bronze_claims, f"{base}/bronze/claims", fmt)

    # Read back from bronze, so silver always builds from the stored copy
    bronze_policies = spark.read.format(fmt).load(f"{base}/bronze/policies")
    bronze_claims = spark.read.format(fmt).load(f"{base}/bronze/claims")

    # ---- Silver ----------------------------------------------------------
    print("Silver: cleaning and checking quality...")
    policies, policies_bad, policies_checked = build_silver_policies(bronze_policies)
    policies = policies.cache()
    claims, claims_bad, claims_checked = build_silver_claims(bronze_claims, policies)

    report = {
        "policies": quality_report(policies_checked, POLICY_RULES),
        "claims": quality_report(claims_checked, CLAIM_RULES),
    }
    report["claims"]["duplicates_removed"] = bronze_claims.count() - report["claims"]["total_rows"]

    rate = report["claims"]["quarantine_rate"]
    if rate > MAX_QUARANTINE_RATE:
        raise RuntimeError(
            f"Quality gate failed: {rate:.1%} of claims quarantined "
            f"(limit {MAX_QUARANTINE_RATE:.0%}). See the report above. Nothing written to gold."
        )

    write(policies, f"{base}/silver/policies", fmt)
    write(claims, f"{base}/silver/claims", fmt)
    write(policies_bad, f"{base}/quarantine/policies", fmt)
    write(claims_bad, f"{base}/quarantine/claims", fmt)

    # ---- Gold ------------------------------------------------------------
    print("Gold: building reporting and feature tables...")
    claims = spark.read.format(fmt).load(f"{base}/silver/claims")
    write(build_claims_monthly(claims, policies), f"{base}/gold/claims_monthly", fmt)
    write(build_fraud_features(claims, policies), f"{base}/gold/fraud_features", fmt)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the claims pipeline.")
    parser.add_argument("--base-path", default="data", help="Folder that contains raw/")
    parser.add_argument("--format", default="parquet", choices=["parquet", "delta"])
    args = parser.parse_args()

    base_path = args.base_path
    if not base_path.startswith("/Volumes") and "://" not in base_path:
        base_path = str(Path(base_path).resolve())

    try:
        report = run(base_path, args.format)
    except RuntimeError as err:
        print(f"\n{err}")
        sys.exit(1)

    print("\nData quality report")
    print(json.dumps(report, indent=2))

    if not base_path.startswith("/Volumes"):
        out = Path(base_path) / "quality_report.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
