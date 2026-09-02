# terraform/databricks.tf
resource "databricks_mws_credentials" "main" {
  account_id       = var.databricks_account_id
  credentials_name = "${var.project_name}-creds-${var.environment}"
  role_arn         = aws_iam_role.databricks_cross_account.arn
}

resource "databricks_mws_storage_configurations" "main" {
  account_id                 = var.databricks_account_id
  storage_configuration_name = "${var.project_name}-storage-${var.environment}"
  bucket_name                = aws_s3_bucket.data_lake.id
}

resource "databricks_mws_networks" "main" {
  account_id   = var.databricks_account_id
  network_name = "${var.project_name}-network-${var.environment}"
  vpc_id       = var.vpc_id
  subnet_ids   = var.private_subnet_ids
  security_group_ids = [
    aws_security_group.msk_sg.id # reuse or create dedicated SG
  ]
}

resource "databricks_mws_workspaces" "main" {
  account_id      = var.databricks_account_id
  workspace_name  = var.databricks_workspace_name
  deployment_name = "${var.project_name}-deployment-${var.environment}"

  aws_region = var.aws_region

  credentials_id           = databricks_mws_credentials.main.credentials_id
  storage_configuration_id = databricks_mws_storage_configurations.main.storage_configuration_id
  network_id               = databricks_mws_networks.main.network_id

  token {
    comment = "Terraform managed token"
  }
}