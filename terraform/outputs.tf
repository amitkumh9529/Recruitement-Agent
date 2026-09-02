# terraform/outputs.tf
output "s3_bucket_arn" {
  description = "ARN of the S3 data lake bucket"
  value       = aws_s3_bucket.data_lake.arn
}

output "s3_bucket_name" {
  description = "Name of the S3 data lake bucket"
  value       = aws_s3_bucket.data_lake.id
}

output "msk_bootstrap_brokers" {
  description = "Bootstrap brokers for MSK cluster"
  value       = aws_msk_cluster.main.bootstrap_brokers
}

output "msk_zookeeper_connect_string" {
  description = "Zookeeper connection string"
  value       = aws_msk_cluster.main.zookeeper_connect_string
}

output "msk_cluster_arn" {
  description = "ARN of the MSK cluster"
  value       = aws_msk_cluster.main.arn
}

output "kms_key_arn" {
  description = "ARN of the KMS key used for encryption"
  value       = aws_kms_key.main.arn
}

output "databricks_workspace_id" {
  description = "ID of the Databricks workspace"
  value       = databricks_mws_workspaces.main.workspace_id
}

output "databricks_workspace_url" {
  description = "URL of the Databricks workspace"
  value       = databricks_mws_workspaces.main.workspace_url
}

output "databricks_workspace_deployment_name" {
  description = "Deployment name of the Databricks workspace"
  value       = databricks_mws_workspaces.main.deployment_name
}