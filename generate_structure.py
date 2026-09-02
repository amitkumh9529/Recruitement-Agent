#!/usr/bin/env python3
"""
Folder & File Generator
Creates the complete project structure for:
enterprise-realtime-transaction-fraud-detection-platform
"""

import os
from pathlib import Path

# Root directory name
ROOT = "Financial transactional datapipeline"

# Complete structure (directories end with /)
STRUCTURE = [
    # Root files
    "README.md",
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "docker-compose.yml",

    # docs/
    "docs/architecture.md",
    "docs/data_dictionary.md",
    "docs/runbook.md",

    # terraform/
    "terraform/versions.tf",
    "terraform/providers.tf",
    "terraform/variables.tf",
    "terraform/outputs.tf",
    "terraform/example.tfvars",
    "terraform/s3.tf",
    "terraform/msk.tf",
    "terraform/iam.tf",
    "terraform/kms.tf",
    "terraform/databricks.tf",

    # ingestion/
    "ingestion/transaction_generator.py",
    "ingestion/entity_generator.py",
    "ingestion/fraud_scenario_generator.py",

    # kafka/
    "kafka/topic_config.yaml",
    "kafka/transaction_schema.json",

    # databricks/
    "databricks/notebooks/bronze_ingestion.py",
    "databricks/notebooks/feature_engineering.py",
    "databricks/notebooks/train_fraud_model.py",
    "databricks/notebooks/model_scoring.py",
    "databricks/jobs/streaming_job.yml",
    "databricks/jobs/feature_job.yml",
    "databricks/jobs/training_job.yml",
    "databricks/feature_store/feature_definitions.py",
    "databricks/model_serving/endpoint_config.json",
    "databricks/sql/catalogs.sql",
    "databricks/sql/schemas.sql",
    "databricks/sql/grants.sql",

    # dbt/
    "dbt/dbt_project.yml",
    "dbt/profiles.yml.example",
    "dbt/packages.yml",
    "dbt/schema.yml",
    "dbt/models/staging/stg_transactions.sql",
    "dbt/models/intermediate/int_transaction_features.sql",
    "dbt/models/intermediate/int_fraud_scoring.sql",
    "dbt/models/silver/transactions.sql",
    "dbt/models/silver/customer_behavior.sql",
    "dbt/models/gold/fact_fraud_transactions.sql",
    "dbt/models/gold/fraud_metrics.sql",
    "dbt/models/marts/fraud_operations.sql",
    "dbt/models/marts/blocked_transactions.sql",
    "dbt/models/marts/customer_risk.sql",
    "dbt/models/marts/model_performance.sql",
    "dbt/tests/assert_valid_amount.sql",
    "dbt/tests/assert_valid_risk_score.sql",
    "dbt/tests/assert_decision_consistency.sql",

    # great_expectations/
    "great_expectations/great_expectations.yml",
    "great_expectations/validate_bronze.py",
    "great_expectations/validate_silver.py",

    # ml/
    "ml/train.py",
    "ml/evaluate.py",
    "ml/model_config.yaml",

    # airflow/
    "airflow/dags/fraud_pipeline_dag.py",
    "airflow/dags/model_retraining_dag.py",

    # streamlit/
    "streamlit/app.py",
    "streamlit/requirements.txt",

    # tests/
    "tests/test_transaction_generator.py",
    "tests/test_data_contract.py",
]


def create_structure(root: str = ROOT) -> None:
    root_path = Path(root)
    root_path.mkdir(exist_ok=True)
    print(f"Created root: {root_path.resolve()}")

    created_dirs = set()
    created_files = 0

    for item in STRUCTURE:
        path = root_path / item

        # Create parent directories
        parent = path.parent
        if parent not in created_dirs:
            parent.mkdir(parents=True, exist_ok=True)
            created_dirs.add(parent)

        # Create empty file
        if not path.exists():
            path.touch()
            created_files += 1
            print(f"  + {path.relative_to(root_path)}")

    print(f"\nDone! Created {len(created_dirs)} directories and {created_files} files.")
    print(f"Project root: {root_path.resolve()}")


if __name__ == "__main__":
    create_structure()