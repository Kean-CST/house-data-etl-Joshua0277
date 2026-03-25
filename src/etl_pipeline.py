"""
House Sale Data ETL Pipeline
============================
Implement the three functions below to complete the ETL pipeline.

Steps:
  1. EXTRACT  – load the CSV into a PySpark DataFrame
  2. TRANSFORM – split the data by neighborhood and save each as a separate CSV
  3. LOAD      – insert each neighborhood DataFrame into its own PostgreSQL table
"""
from __future__ import annotations

import csv  # noqa: F401
import os  # noqa: F401
import shutil
from pathlib import Path

from dotenv import load_dotenv  # noqa: F401
from pyspark.sql import DataFrame, SparkSession  # noqa: F401
from pyspark.sql import functions as F  # noqa: F401

# ── Predefined constants (do not modify) ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

NEIGHBORHOODS = [
    "Downtown", "Green Valley", "Hillcrest", "Lakeside", "Maple Heights",
    "Oakwood", "Old Town", "Riverside", "Suburban Park", "University District",
]

OUTPUT_DIR   = ROOT / "output" / "by_neighborhood"
OUTPUT_FILES = {hood: OUTPUT_DIR / f"{hood.replace(' ', '_').lower()}.csv" for hood in NEIGHBORHOODS}

PG_TABLES = {hood: f"public.{hood.replace(' ', '_').lower()}" for hood in NEIGHBORHOODS}

PG_COLUMN_SCHEMA = (
    "house_id TEXT, neighborhood TEXT, price INTEGER, square_feet INTEGER, "
    "num_bedrooms INTEGER, num_bathrooms INTEGER, house_age INTEGER, "
    "garage_spaces INTEGER, lot_size_acres NUMERIC(6,2), has_pool BOOLEAN, "
    "recently_renovated BOOLEAN, energy_rating TEXT, location_score INTEGER, "
    "school_rating INTEGER, crime_rate INTEGER, "
    "distance_downtown_miles NUMERIC(6,2), sale_date DATE, days_on_market INTEGER"
)


def extract(spark: SparkSession, csv_path: str) -> DataFrame:
    """Load the CSV dataset into a PySpark DataFrame with correct data types."""
    return spark.read.csv(csv_path, header=True, inferSchema=True)


def transform(df: DataFrame) -> dict[str, DataFrame]:
    """Split the data by neighborhood and save each as a separate CSV file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    partitions: dict[str, DataFrame] = {}
    for hood in NEIGHBORHOODS:
        # 1. Ensure ordering
        hood_df = df.filter(F.col("neighborhood") == hood).orderBy("house_id")
        out_path = OUTPUT_FILES[hood]

        # 2. Create the specialized DataFrame for CSV writing
        csv_df = hood_df
        
        # --- Format Boolean Columns ---
        for col_name, col_type in df.dtypes:
            if col_type == 'boolean':
                csv_df = csv_df.withColumn(col_name, F.initcap(F.col(col_name).cast("string")))

        # --- Format Date Column ---
        # We must explicitly convert the M/D/YY string into a real Date, 
        # then format it exactly as YYYY-MM-DD for the strict test script.
        if "sale_date" in csv_df.columns:
            csv_df = csv_df.withColumn(
                "sale_date", 
                F.date_format(F.to_date(F.col("sale_date"), "M/d/yy"), "yyyy-MM-dd")
            )

        # 3. Write to temporary directory
        tmp_dir = OUTPUT_DIR / f"_tmp_{hood.replace(' ', '_').lower()}"
        csv_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(str(tmp_dir))

        # 4. Move and cleanup
        part_files = list(tmp_dir.glob("part-*.csv"))
        if part_files:
            shutil.move(str(part_files[0]), str(out_path))
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

        # 5. Keep the original for Postgres
        partitions[hood] = hood_df

    return partitions


def load(partitions: dict[str, DataFrame], jdbc_url: str, pg_props: dict) -> None:
    """Insert each neighborhood dataset into its own PostgreSQL table."""
    for hood, hood_df in partitions.items():
        table_name = PG_TABLES[hood]
        
        # Write the DataFrame to the PostgreSQL table
        hood_df.write.jdbc(
            url=jdbc_url,
            table=table_name,
            mode="overwrite",
            properties=pg_props
        )


# ── Main (do not modify) ───────────────────────────────────────────────────────
def main() -> None:
    load_dotenv(ROOT / ".env")

    jdbc_url = (
        f"jdbc:postgresql://{os.getenv('PG_HOST', 'localhost')}:"
        f"{os.getenv('PG_PORT', '5432')}/{os.environ['PG_DATABASE']}"
    )
    pg_props = {
        "user":     os.environ["PG_USER"],
        "password": os.getenv("PG_PASSWORD", ""),
        "driver":   "org.postgresql.Driver",
    }
    csv_path = str(ROOT / os.getenv("DATASET_DIR", "dataset") / os.getenv("DATASET_FILE", "historical_purchases.csv"))

    spark = (
        SparkSession.builder.appName("HouseSaleETL")
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df         = extract(spark, csv_path)
    partitions = transform(df)
    load(partitions, jdbc_url, pg_props)

    spark.stop()


if __name__ == "__main__":
    main()
