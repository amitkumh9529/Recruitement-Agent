# databricks/feature_store/feature_definitions.py
"""
Feature definitions for the fraud detection feature store.
This file is used to define feature computation functions that can be registered
with Databricks Feature Store. It separates feature logic for reuse.
"""

from databricks.feature_store import FeatureStoreClient
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, avg, count, stddev, max, min, window, lag, unix_timestamp
from pyspark.sql.window import Window


def compute_customer_features(df: DataFrame) -> DataFrame:
    """
    Compute customer-level aggregate features from raw transactions.

    Args:
        df: DataFrame with raw transaction columns.

    Returns:
        DataFrame with additional feature columns.
    """
    # Convert timestamp to timestamp type
    df = df.withColumn("event_ts", (col("timestamp") / 1000).cast("timestamp"))

    # Window specification for rolling aggregates (24 hours)
    rolling_window = (
        Window.partitionBy("customer_id")
        .orderBy(col("event_ts").cast("long"))
        .rangeBetween(-86400, 0)  # 24 hours in seconds
    )

    df_features = (
        df
        .withColumn("txn_count_24h", count("transaction_id").over(rolling_window))
        .withColumn("avg_amount_24h", avg("amount").over(rolling_window))
        .withColumn("stddev_amount_24h", stddev("amount").over(rolling_window))
        .withColumn("max_amount_24h", max("amount").over(rolling_window))
        .withColumn("min_amount_24h", min("amount").over(rolling_window))
        .withColumn("unique_merchants_24h", 
                    countDistinct("merchant_id").over(rolling_window))
        .withColumn("unique_countries_24h", 
                    countDistinct("location.country").over(rolling_window))
        .withColumn("prev_txn_ts", lag("event_ts", 1).over(
            Window.partitionBy("customer_id").orderBy("event_ts")))
        .withColumn("time_since_last_txn_sec",
                    (unix_timestamp(col("event_ts")) - unix_timestamp(col("prev_txn_ts"))))
        .withColumn("amount_zscore",
                    (col("amount") - avg("amount").over(Window.partitionBy("customer_id"))) /
                    (stddev("amount").over(Window.partitionBy("customer_id")) + 0.001))
    )

    return df_features