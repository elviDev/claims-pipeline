"""
Run the whole pipeline: raw CSV -> bronze -> silver (+ quarantine) -> gold.

Locally / in Docker (parquet folders under data/):
    python -m pipeline.run_pipeline

On Databricks (Unity Catalog tables workspace.claims.bronze_claims, ...,
raw CSVs in the volume /Volumes/workspace/claims/raw):
    python -m pipeline.run_pipeline --tables workspace.claims
or from a notebook: run(Storage("table", "workspace.claims"), "/Volumes/workspace/claims/raw", spark)
"""

import argparse
import json
import sys
from pathlib import Path

from pyspark.sql import SparkSession

from pipeline.bronze import read_raw_csv
from pipeline.gold import build_claims_monthly, build_fraud_features
from pipeline.quality import quality_report
from pipeline.silver import CLAIM_RULES, POLICY_RULES, build_silver_claims, build_silver_policies
from pipeline.spark import get_spark
from pipeline.storage import Storage

# Quality gate: if more than this share of claims fails the rules, stop.
# Something is badly wrong with the source, and loading it would poison
# every report and model downstream.
MAX_QUARANTINE_RATE = 0.05


def run(storage: Storage, raw_dir: str, spark: SparkSession | None = None) -> dict:
    """
    Run every layer. Returns the data quality report.

    `spark` can be passed in: a Databricks notebook already has one.
    """
    spark = spark or get_spark()
    raw = raw_dir.rstrip("/")

    # ---- Bronze ----------------------------------------------------------
    print("Bronze: loading raw files...")
    storage.write(read_raw_csv(spark, f"{raw}/policies.csv"), "bronze", "policies")
    storage.write(read_raw_csv(spark, f"{raw}/claims.csv"), "bronze", "claims")

    # Every layer builds from the stored copy of the layer before it, not from
    # something held in memory. So any layer can be rebuilt on its own, and
    # nothing needs .cache() (which Databricks serverless doesn't support).
    bronze_policies = storage.read(spark, "bronze", "policies")
    bronze_claims = storage.read(spark, "bronze", "claims")

    # ---- Silver ----------------------------------------------------------
    print("Silver: cleaning and checking quality...")
    policies, policies_bad, policies_checked = build_silver_policies(bronze_policies)
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

    storage.write(policies, "silver", "policies")
    storage.write(claims, "silver", "claims")
    storage.write(policies_bad, "quarantine", "policies")
    storage.write(claims_bad, "quarantine", "claims")

    # ---- Gold ------------------------------------------------------------
    print("Gold: building reporting and feature tables...")
    claims = storage.read(spark, "silver", "claims")
    policies = storage.read(spark, "silver", "policies")
    storage.write(build_claims_monthly(claims, policies), "gold", "claims_monthly")
    storage.write(build_fraud_features(claims, policies), "gold", "fraud_features")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the claims pipeline.")
    parser.add_argument("--base-path", default="data", help="Path mode: folder for bronze/, silver/, ...")
    parser.add_argument("--format", default="parquet", choices=["parquet", "delta"], help="Path mode only")
    parser.add_argument("--tables", metavar="CATALOG.SCHEMA",
                        help="Table mode: write Unity Catalog tables here, e.g. workspace.claims")
    parser.add_argument("--raw-dir", help="Folder with policies.csv and claims.csv "
                                          "(default: <base-path>/raw, or /Volumes/<catalog>/<schema>/raw)")
    args = parser.parse_args()

    if args.tables:
        storage = Storage("table", args.tables)
        raw_dir = args.raw_dir or "/Volumes/" + args.tables.replace(".", "/") + "/raw"
    else:
        base_path = args.base_path
        if "://" not in base_path:
            base_path = str(Path(base_path).resolve())
        storage = Storage("path", base_path, args.format)
        raw_dir = args.raw_dir or f"{base_path}/raw"

    try:
        report = run(storage, raw_dir)
    except RuntimeError as err:
        print(f"\n{err}")
        sys.exit(1)

    print("\nData quality report")
    print(json.dumps(report, indent=2))

    # Locally, also save the report next to the data. In table mode the
    # notebook shows it instead.
    if storage.mode == "path" and "://" not in storage.base:
        out = Path(storage.base) / "quality_report.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
