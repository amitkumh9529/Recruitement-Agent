enterprise-realtime-transaction-fraud-detection-platform/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── docker-compose.yml
│
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   └── runbook.md
│
├── terraform/
│   ├── versions.tf
│   ├── providers.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── example.tfvars
│   ├── s3.tf
│   ├── msk.tf
│   ├── iam.tf
│   ├── kms.tf
│   └── databricks.tf
│
├── ingestion/
│   ├── transaction_generator.py
│   ├── entity_generator.py
│   └── fraud_scenario_generator.py
│
├── kafka/
│   ├── topic_config.yaml
│   └── transaction_schema.json
│
├── databricks/
│   │
│   ├── notebooks/
│   │   ├── bronze_ingestion.py
│   │   ├── feature_engineering.py
│   │   ├── train_fraud_model.py
│   │   └── model_scoring.py
│   │
│   ├── jobs/
│   │   ├── streaming_job.yml
│   │   ├── feature_job.yml
│   │   └── training_job.yml
│   │
│   ├── feature_store/
│   │   └── feature_definitions.py
│   │
│   ├── model_serving/
│   │   └── endpoint_config.json
│   │
│   └── sql/
│       ├── catalogs.sql
│       ├── schemas.sql
│       └── grants.sql
│
├── dbt/
│   ├── dbt_project.yml
│   ├── profiles.yml.example
│   ├── packages.yml
│   ├── schema.yml
│   │
│   ├── models/
│   │   ├── staging/
│   │   │   └── stg_transactions.sql
│   │   │
│   │   ├── intermediate/
│   │   │   ├── int_transaction_features.sql
│   │   │   └── int_fraud_scoring.sql
│   │   │
│   │   ├── silver/
│   │   │   ├── transactions.sql
│   │   │   └── customer_behavior.sql
│   │   │
│   │   ├── gold/
│   │   │   ├── fact_fraud_transactions.sql
│   │   │   └── fraud_metrics.sql
│   │   │
│   │   └── marts/
│   │       ├── fraud_operations.sql
│   │       ├── blocked_transactions.sql
│   │       ├── customer_risk.sql
│   │       └── model_performance.sql
│   │
│   └── tests/
│       ├── assert_valid_amount.sql
│       ├── assert_valid_risk_score.sql
│       └── assert_decision_consistency.sql
│
├── great_expectations/
│   ├── great_expectations.yml
│   ├── validate_bronze.py
│   └── validate_silver.py
│
├── ml/
│   ├── train.py
│   ├── evaluate.py
│   └── model_config.yaml
│
├── airflow/
│   └── dags/
│       ├── fraud_pipeline_dag.py
│       └── model_retraining_dag.py
│
├── streamlit/
│   ├── app.py
│   └── requirements.txt
│
└── tests/
    ├── test_transaction_generator.py
    └── test_data_contract.py