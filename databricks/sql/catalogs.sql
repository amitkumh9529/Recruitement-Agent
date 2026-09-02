-- databricks/sql/catalogs.sql
-- Create Unity Catalog catalogs and schemas for fraud detection.

CREATE CATALOG IF NOT EXISTS fraud_detection;

-- Schemas for medallion architecture
CREATE SCHEMA IF NOT EXISTS fraud_detection.bronze;
CREATE SCHEMA IF NOT EXISTS fraud_detection.silver;
CREATE SCHEMA IF NOT EXISTS fraud_detection.gold;
CREATE SCHEMA IF NOT EXISTS fraud_detection.feature_store;