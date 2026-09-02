# terraform/example.tfvars
aws_region                 = "us-east-1"
environment                = "dev"
project_name               = "fraud-detection"
vpc_id                     = "vpc-0123456789abcdef0"
private_subnet_ids         = ["subnet-0123456789abcdef1", "subnet-0123456789abcdef2"]
public_subnet_ids          = ["subnet-0123456789abcdef3", "subnet-0123456789abcdef4"]
msk_kafka_version          = "3.5.1"
msk_instance_type          = "kafka.m5.large"
msk_broker_count           = 3
msk_storage_gb             = 100
s3_bucket_name             = "fraud-detection-data-lake"
databricks_account_id      = "12345678-1234-1234-1234-123456789012"
databricks_workspace_url   = "https://dbc-12345678-1234.cloud.databricks.com"
databricks_token           = "dapi1234567890abcdef"
databricks_workspace_name  = "fraud-detection-workspace"
tags = {
  Project     = "FraudDetection"
  Environment = "dev"
  Owner       = "DataEng"
}