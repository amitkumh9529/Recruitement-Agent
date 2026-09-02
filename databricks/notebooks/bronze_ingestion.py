# databricks/notebooks/bronze_ingestion.py
# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Ingestion: Kafka to Delta Lake
# MAGIC Reads raw transactions from Amazon MSK and writes to Bronze Delta table.

# COMMAND ----------

# DBTITLE 1,Configuration
kafka_bootstrap_servers = "b-1.msk-cluster.abc123.kafka.us-east-1.amazonaws.com:9094,b-2.msk-cluster.abc123.kafka.us-east-1.amazonaws.com:9094"
kafka_topic = "transactions.raw"
bronze_table = "fraud_detection.bronze.transactions"
checkpoint_path = "/mnt/delta/checkpoints/bronze_transactions"
starting_offsets = "latest"  # or "earliest"

# COMMAND ----------

# DBTITLE 1,Define Schema
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, LongType, BooleanType, IntegerType,
    ArrayType, MapType
)

# Schema matching the Avro/JSON structure from the producer
transaction_schema = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("merchant_id", StringType(), False),
    StructField("merchant_category", StringType(), False),
    StructField("amount", DoubleType(), False),
    StructField("currency", StringType(), False),
    StructField("timestamp", LongType(), False),
    StructField("location", StructType([
        StructField("latitude", DoubleType(), False),
        StructField("longitude", DoubleType(), False),
        StructField("country", StringType(), False),
        StructField("city", StringType(), False)
    ]), False),
    StructField("device_id", StringType(), True),
    StructField("ip_address", StringType(), True),
    StructField("card_present", BooleanType(), False),
    StructField("card_type", StringType(), True),
    StructField("account_age_days", IntegerType(), True),
    StructField("transaction_type", StringType(), False),
    StructField("is_fraud", BooleanType(), True)  # ground truth, may be null in real-time
])

# COMMAND ----------

# DBTITLE 1,Read Stream from Kafka
stream_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
    .option("subscribe", kafka_topic)
    .option("startingOffsets", starting_offsets)
    .option("failOnDataLoss", "false")
    .load()
)

# COMMAND ----------

# DBTITLE 1,Parse JSON and Apply Schema
from pyspark.sql.functions import from_json, col

parsed_df = (
    stream_df
    .selectExpr("CAST(value AS STRING) as json")
    .select(from_json(col("json"), transaction_schema).alias("data"))
    .select("data.*")
)

# COMMAND ----------

# DBTITLE 1,Write to Delta
(
    parsed_df.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", checkpoint_path)
    .option("mergeSchema", "true")
    .table(bronze_table)
)