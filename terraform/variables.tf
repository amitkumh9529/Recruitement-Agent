# terraform/variables.tf
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (e.g., dev, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "fraud-detection"
}

variable "vpc_id" {
  description = "VPC ID where MSK and Databricks will be deployed"
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for MSK and Databricks"
  type        = list(string)
}

variable "public_subnet_ids" {
  description = "List of public subnet IDs for NAT/load balancers"
  type        = list(string)
}

variable "msk_kafka_version" {
  description = "Kafka version for MSK"
  type        = string
  default     = "3.5.1"
}

variable "msk_instance_type" {
  description = "MSK broker instance type"
  type        = string
  default     = "kafka.m5.large"
}

variable "msk_broker_count" {
  description = "Number of brokers in MSK cluster"
  type        = number
  default     = 3
}

variable "msk_storage_gb" {
  description = "Storage per broker in GB"
  type        = number
  default     = 100
}

variable "s3_bucket_name" {
  description = "Base name for S3 data lake bucket (will be suffixed with random string)"
  type        = string
  default     = "fraud-detection-data-lake"
}

variable "databricks_account_id" {
  description = "Databricks account ID for workspace creation"
  type        = string
}

variable "databricks_workspace_url" {
  description = "Databricks workspace URL (used for provider authentication)"
  type        = string
}

variable "databricks_token" {
  description = "Databricks personal access token"
  type        = string
  sensitive   = true
}

variable "databricks_workspace_name" {
  description = "Name for the Databricks workspace"
  type        = string
  default     = "fraud-detection-workspace"
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "FraudDetection"
    Environment = "dev"
  }
}