# databricks/notebooks/model_scoring.py
# Databricks notebook source
# MAGIC %md
# MAGIC # Real-Time Model Scoring
# MAGIC Reads streaming transactions from Kafka, computes features using Feature Store, scores with MLflow model, and writes results to Silver (scored).

# COMMAND ----------

# DBTITLE 1,Configuration
from databricks.feature_store import FeatureStoreClient
import mlflow
from pyspark.sql.functions import from_json, col, struct
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType, IntegerType

kafka_bootstrap_servers = "b-1.msk-cluster.abc123.kafka.us-east-1.amazonaws.com:9094"
kafka_topic = "transactions.raw"
model_name = "fraud_xgboost"
scored_table = "fraud_detection.silver.scored_transactions"
checkpoint_path = "/mnt/delta/checkpoints/scoring"

fs = FeatureStoreClient()

# COMMAND ----------

# DBTITLE 1,Load Model from MLflow
model_uri = f"models:/{model_name}/Production"
model = mlflow.xgboost.load_model(model_uri)

# COMMAND ----------

# DBTITLE 1,Define Schema and Read Stream
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
    StructField("is_fraud", BooleanType(), True)
])

stream_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
    .option("subscribe", kafka_topic)
    .option("startingOffsets", "latest")
    .option("failOnDataLoss", "false")
    .load()
    .selectExpr("CAST(value AS STRING) as json")
    .select(from_json(col("json"), transaction_schema).alias("data"))
    .select("data.*")
)

# COMMAND ----------

# DBTITLE 1,Compute Features (Simplified: using Feature Store lookup)
# For real-time, we would use FeatureStoreClient to look up customer aggregates.
# For simplicity, we'll compute basic features on the fly (window functions not possible in streaming easily)
# In production, you'd maintain precomputed features in a key-value store like Redis.
# Here we demonstrate using a static feature table and join with batch updates.

# We'll assume the Feature Store has latest customer stats, and we join on customer_id.
# For the sake of simplicity, we'll just use the raw columns as features.
# The model expects feature columns: amount, account_age_days, card_present, transaction_type_encoded, etc.
# We'll create dummy features for demonstration; replace with actual feature engineering.

from pyspark.sql.functions import when, udf
from pyspark.sql.types import FloatType

# Dummy feature columns matching model input
def score_partition(partition):
    import pandas as pd
    import mlflow
    import xgboost as xgb
    # Load model within partition (or use broadcast)
    model = mlflow.xgboost.load_model(model_uri)
    pdf = pd.DataFrame(list(partition))
    # Ensure columns match training features
    # For demo, use only amount and account_age_days
    X = pdf[["amount", "account_age_days"]]
    dmatrix = xgb.DMatrix(X)
    preds = model.predict(dmatrix)
    pdf["fraud_probability"] = preds
    pdf["fraud_prediction"] = (preds > 0.5).astype(int)
    return pdf

# Apply scoring using foreachBatch to avoid UDF complexity
def score_batch(df, epoch_id):
    # Convert to pandas for simplicity in batch, but for production use vectorized UDF
    if df.count() > 0:
        pdf = df.toPandas()
        X = pdf[["amount", "account_age_days"]]
        dmatrix = xgb.DMatrix(X)
        preds = model.predict(dmatrix)
        pdf["fraud_probability"] = preds
        pdf["fraud_prediction"] = (preds > 0.5).astype(int)
        scored_spark_df = spark.createDataFrame(pdf)
        scored_spark_df.write.format("delta").mode("append").saveAsTable(scored_table)
    else:
        pass

# COMMAND ----------

# DBTITLE 1,Write Stream with Scoring
# Use foreachBatch to apply batch scoring
(
    stream_df.writeStream
    .foreachBatch(score_batch)
    .outputMode("append")
    .option("checkpointLocation", checkpoint_path)
    .start()
)