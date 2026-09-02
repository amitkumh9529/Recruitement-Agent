-- databricks/sql/grants.sql
-- Grants for Unity Catalog
-- Assumes roles/groups exist.

-- Grant usage on catalog to data engineers and analysts
GRANT USAGE ON CATALOG fraud_detection TO `fraud_data_engineers`;
GRANT USAGE ON CATALOG fraud_detection TO `fraud_analysts`;

-- Grant schema-level permissions
GRANT USAGE ON SCHEMA fraud_detection.bronze TO `fraud_data_engineers`;
GRANT USAGE ON SCHEMA fraud_detection.silver TO `fraud_data_engineers`;
GRANT USAGE ON SCHEMA fraud_detection.gold TO `fraud_data_engineers`;
GRANT USAGE ON SCHEMA fraud_detection.gold TO `fraud_analysts`;

-- Grant read/write on bronze for data engineers
GRANT SELECT, MODIFY, CREATE ON SCHEMA fraud_detection.bronze TO `fraud_data_engineers`;

-- Grant read/write on silver for data engineers
GRANT SELECT, MODIFY, CREATE ON SCHEMA fraud_detection.silver TO `fraud_data_engineers`;

-- Grant read on gold for analysts
GRANT SELECT ON SCHEMA fraud_detection.gold TO `fraud_analysts`;

-- Column-level access control example: restrict PII in silver
GRANT SELECT ON TABLE fraud_detection.silver.transactions_features TO `fraud_analysts`;  -- no PII columns
-- Alternatively, use dynamic views or column masks for more granular control.