# databricks/notebooks/train_fraud_model.py
# Databricks notebook source
# MAGIC %md
# MAGIC # Train Fraud Detection Model
# MAGIC Trains an XGBoost classifier using features from Feature Store and logs to MLflow.

# COMMAND ----------

# DBTITLE 1,Imports and Config
import mlflow
import mlflow.xgboost
from databricks.feature_store import FeatureStoreClient
from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import pandas as pd

fs = FeatureStoreClient()
feature_table = "fraud_detection.feature_store.customer_features"
model_name = "fraud_xgboost"
experiment_name = "/Shared/fraud-detection"

mlflow.set_experiment(experiment_name)

# COMMAND ----------

# DBTITLE 1,Load Training Data from Feature Store
training_df = fs.read_table(feature_table)

# For training, we need labeled data (is_fraud is not null)
training_df = training_df.filter(col("is_fraud").isNotNull())

# Convert to pandas for XGBoost
pandas_df = training_df.toPandas()

# Define features and label
feature_cols = [c for c in pandas_df.columns if c not in ["transaction_id", "customer_id", "merchant_id", "timestamp", "is_fraud"]]
X = pandas_df[feature_cols]
y = pandas_df["is_fraud"].astype(int)

# COMMAND ----------

# DBTITLE 1,Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# DBTITLE 1,Train XGBoost Model
with mlflow.start_run() as run:
    # Define model parameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, "test")], early_stopping_rounds=10)
    
    # Evaluate
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Log parameters and metrics
    mlflow.log_params(params)
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    })
    
    # Log model
    mlflow.xgboost.log_model(model, "model", registered_model_name=model_name)
    
    print(f"Model logged with run_id: {run.info.run_id}")
    print(f"Metrics: accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, roc_auc={roc_auc:.4f}")