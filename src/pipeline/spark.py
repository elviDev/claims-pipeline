"""One place to create the Spark session, so every step uses the same settings."""

import os

from pyspark.sql import SparkSession


def on_databricks() -> bool:
    """Databricks sets this variable on every cluster and serverless environment."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_spark(app_name: str = "claims-pipeline") -> SparkSession:
    """
    Return a Spark session.

    On Databricks a session already exists, so we return it untouched.
    Serverless runs on Spark Connect: the notebook talks to a Spark service
    that Databricks manages, so settings like .master() or the number of
    shuffle partitions aren't ours to set there (and trying can fail).

    Locally (or in Docker) this starts a small Spark running inside this process.
    """
    if on_databricks():
        return SparkSession.builder.getOrCreate()

    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")                               # use all CPU cores on this machine
        .config("spark.sql.shuffle.partitions", "4")      # default is 200, far too many for small local data
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
