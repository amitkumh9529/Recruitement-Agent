# databricks/notebooks/feature_engineering.py
# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering
# MAGIC Reads Bronze transactions, computes features, and writes to Silver and Feature Store.

# COMMAND ----------

# DBTITLE 1,Configuration
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

bronze_table = "fraud_detection.bronze.transactions"
silver_table = "fraud_detection.silver.transactions_features"
feature_table = "fraud_detection.feature_store.customer_features"

# COMMAND ----------

# DBTITLE 1,Read Bronze Data (Batch Mode for Scheduled Job)
df = spark.table(bronze_table)

# COMMAND ----------

# DBTITLE 1,Compute Features
from pyspark.sql.functions import (
    col, avg, sum, count, when, unix_timestamp, from_unixtime, window,
    stddev, max, min, mean
)
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# Convert timestamp milliseconds to seconds for time-based operations
df = df.withColumn("event_ts", (col("timestamp") / 1000).cast("timestamp"))

# Customer-level aggregations over a sliding window (e.g., last 24 hours)
window_spec = Window.partitionBy("customer_id").orderBy(col("event_ts").cast("long")).rangeBetween(-86400, 0)

df_features = (
    df
    .withColumn("txn_count_24h", count("transaction_id").over(window_spec))
    .withColumn("avg_amount_24h", avg("amount").over(window_spec))
    .withColumn("stddev_amount_24h", stddev("amount").over(window_spec))
    .withColumn("max_amount_24h", max("amount").over(window_spec))
    .withColumn("min_amount_24h", min("amount").over(window_spec))
    .withColumn("unique_merchants_24h", F.size(F.collect_set("merchant_id").over(window_spec)))
    .withColumn("unique_countries_24h", F.size(F.collect_set("location.country").over(window_spec)))
    # Time since last transaction
    .withColumn("prev_txn_ts", F.lag("event_ts", 1).over(Window.partitionBy("customer_id").orderBy("event_ts")))
    .withColumn("time_since_last_txn_sec", 
                (unix_timestamp(col("event_ts")) - unix_timestamp(col("prev_txn_ts"))))
    # Amount relative to customer's average
    .withColumn("amount_zscore", 
                (col("amount") - avg("amount").over(Window.partitionBy("customer_id"))) / 
                (stddev("amount").over(Window.partitionBy("customer_id")) + 0.001))
)

# Select relevant features
feature_cols = [
    "transaction_id",
    "customer_id",
    "merchant_id",
    "amount",
    "timestamp",
    "transaction_type",
    "card_present",
    "account_age_days",
    "txn_count_24h",
    "avg_amount_24h",
    "stddev_amount_24h",
    "max_amount_24h",
    "min_amount_24h",
    "unique_merchants_24h",
    "unique_countries_24h",
    "time_since_last_txn_sec",
    "amount_zscore",
    "is_fraud",
]

df_features = df_features.select(*feature_cols)

# COMMAND ----------

# DBTITLE 1,Write to Silver Delta Table
df_features.write.format("delta").mode("overwrite").saveAsTable(silver_table)

# COMMAND ----------

# DBTITLE 1,Create/Update Feature Store Table
# Define feature table schema and create if not exists
# For simplicity, we'll write the features as a feature table.
# In practice, you may separate customer-level and transaction-level features.
fs.create_table(
    name=feature_table,
    primary_keys=["customer_id", "timestamp"],
    df=df_features,
    description="Customer transaction features for fraud detection",
    tags={"source": "bronze_transactions"}
)