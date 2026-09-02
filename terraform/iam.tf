# terraform/iam.tf
# IAM role for Databricks cross-account access
data "aws_caller_identity" "current" {}

resource "aws_iam_role" "databricks_cross_account" {
  name = "${var.project_name}-databricks-cross-account-${var.environment}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${var.databricks_account_id}:root"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "sts:ExternalId" = var.databricks_account_id
          }
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "databricks_cross_account_s3" {
  role       = aws_iam_role.databricks_cross_account.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess" # scope down in production
}

resource "aws_iam_role_policy_attachment" "databricks_cross_account_kms" {
  role       = aws_iam_role.databricks_cross_account.name
  policy_arn = aws_iam_policy.databricks_kms_policy.arn
}

resource "aws_iam_policy" "databricks_kms_policy" {
  name        = "${var.project_name}-databricks-kms-${var.environment}"
  description = "Policy for Databricks to use KMS key"
  policy      = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = [aws_kms_key.main.arn]
      }
    ]
  })
}

# IAM role for MSK to use KMS and CloudWatch
resource "aws_iam_role" "msk_role" {
  name = "${var.project_name}-msk-role-${var.environment}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "kafka.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "msk_kms" {
  role       = aws_iam_role.msk_role.name
  policy_arn = aws_iam_policy.msk_kms_policy.arn
}

resource "aws_iam_policy" "msk_kms_policy" {
  name        = "${var.project_name}-msk-kms-${var.environment}"
  description = "Policy for MSK to use KMS key"
  policy      = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = [aws_kms_key.main.arn]
      }
    ]
  })
}