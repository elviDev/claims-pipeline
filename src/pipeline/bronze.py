"""
BRONZE layer: land the raw data exactly as it arrived.

Rules for bronze:
  * No cleaning, no fixing, no type conversion. Every column is read as text.
  * Add metadata columns: when it was loaded and which file it came from.

Why keep the raw data untouched? If a bug in the silver logic is found next
month, we can fix the code and rebuild silver from bronze. Without a faithful
copy of the raw data, that mistake would be permanent.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def read_raw_csv(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", False)   # keep everything as strings in bronze
        .option("multiLine", True)      # descriptions can contain commas and quotes
        .option("escape", '"')
        .csv(path)
        .withColumn("_ingested_at", F.current_timestamp())
        # _metadata is a hidden column Spark adds to every file it reads.
        # (Works on Databricks too, where the older input_file_name() is blocked.)
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )
