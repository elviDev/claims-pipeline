"""One place to create the Spark session, so every step uses the same settings."""

from pyspark.sql import SparkSession


def get_spark(app_name: str = "claims-pipeline") -> SparkSession:
    """
    Return a Spark session.

    On Databricks a session already exists, and getOrCreate() simply returns it.
    Locally (or in Docker) this starts a small Spark running inside this process.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")                               # use all CPU cores on this machine
        .config("spark.sql.shuffle.partitions", "4")      # default is 200, far too many for small local data
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
