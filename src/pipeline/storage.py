"""
Where each layer of the pipeline is stored: folders locally, tables on Databricks.

The pipeline code only ever says "write this DataFrame as silver / claims".
Storage decides what that means:

  path mode  (local default)   data/silver/claims             a parquet folder
  table mode (Databricks)      workspace.claims.silver_claims  a Unity Catalog table

On Databricks, tables are the natural home: Unity Catalog manages the files,
permissions and lineage, and the table format is Delta (parquet files plus a
transaction log, which gives all-or-nothing writes and a version history).
The medallion layers become a naming convention: bronze_*, silver_*, gold_*.
"""

from dataclasses import dataclass
from typing import Literal

from pyspark.sql import DataFrame, SparkSession


@dataclass(frozen=True)
class Storage:
    mode: Literal["path", "table"]
    # path mode: the folder that holds bronze/, silver/, ... (e.g. "data")
    # table mode: "catalog.schema" (e.g. "workspace.claims")
    base: str
    fmt: str = "parquet"    # path mode only. Tables in Unity Catalog are Delta.

    def __post_init__(self):
        if self.mode == "table" and self.base.count(".") != 1:
            raise ValueError(f"Table mode needs base as 'catalog.schema', got {self.base!r}")

    def location(self, layer: str, name: str) -> str:
        """
        Where one table lives.
            path:  data/silver/claims
            table: workspace.claims.silver_claims
        """
        if self.mode == "table":
            return f"{self.base}.{layer}_{name}"
        return f"{self.base.rstrip('/')}/{layer}/{name}"

    def write(self, df: DataFrame, layer: str, name: str) -> None:
        # overwrite: each run rebuilds the layer from the one before it, so a
        # rerun gives the same result instead of duplicating rows
        writer = df.write.mode("overwrite")
        if self.mode == "table":
            # overwriteSchema: if a column is added to the code, the table's
            # schema is replaced too. Without it, Delta refuses the write.
            writer.option("overwriteSchema", "true").saveAsTable(self.location(layer, name))
        else:
            writer.format(self.fmt).save(self.location(layer, name))

    def read(self, spark: SparkSession, layer: str, name: str) -> DataFrame:
        if self.mode == "table":
            return spark.table(self.location(layer, name))
        return spark.read.format(self.fmt).load(self.location(layer, name))
