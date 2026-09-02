# terraform/kms.tf
resource "aws_kms_key" "main" {
  description             = "KMS key for encryption of data lake and MSK"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = var.tags
}

resource "aws_kms_alias" "main" {
  name          = "alias/${var.project_name}-${var.environment}"
  target_key_id = aws_kms_key.main.key_id
}