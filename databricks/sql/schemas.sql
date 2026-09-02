-- databricks/sql/schemas.sql
-- Additional schema-level configurations (if needed)
-- This script is optional; catalogs.sql already creates schemas.
-- Use this for schema-level properties like location, comments, etc.

COMMENT ON SCHEMA fraud_detection.bronze IS 'Raw ingested data from Kafka';
COMMENT ON SCHEMA fraud_detection.silver IS 'Cleaned and feature-engineered data';
COMMENT ON SCHEMA fraud_detection.gold IS 'Aggregated and business-ready data';
COMMENT ON SCHEMA fraud_detection.feature_store IS 'Databricks Feature Store tables';