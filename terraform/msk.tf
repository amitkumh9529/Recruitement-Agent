# terraform/msk.tf
resource "aws_security_group" "msk_sg" {
  name_prefix = "${var.project_name}-msk-"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 9094
    to_port     = 9094
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # restrict in production
    description = "Kafka TLS"
  }

  ingress {
    from_port   = 2181
    to_port     = 2181
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # restrict in production
    description = "Zookeeper"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

resource "aws_msk_configuration" "main" {
  kafka_versions = [var.msk_kafka_version]
  name           = "${var.project_name}-config-${var.environment}"

  server_properties = <<PROPERTIES
auto.create.topics.enable = false
default.replication.factor = 3
min.insync.replicas = 2
num.partitions = 3
log.retention.hours = 168
PROPERTIES
}

resource "aws_msk_cluster" "main" {
  cluster_name           = "${var.project_name}-msk-${var.environment}"
  kafka_version          = var.msk_kafka_version
  number_of_broker_nodes = var.msk_broker_count

  broker_node_group_info {
    instance_type   = var.msk_instance_type
    client_subnets  = var.private_subnet_ids
    security_groups = [aws_security_group.msk_sg.id]
    storage_info {
      ebs_storage_info {
        volume_size = var.msk_storage_gb
      }
    }
  }

  configuration_info {
    arn      = aws_msk_configuration.main.arn
    revision = aws_msk_configuration.main.latest_revision
  }

  encryption_info {
    encryption_in_transit {
      client_broker = "TLS"
      in_cluster    = true
    }
    encryption_at_rest_kms_key_arn = aws_kms_key.main.arn
  }

  open_monitoring {
    prometheus {
      jmx_exporter {
        enabled_in_broker = true
      }
      node_exporter {
        enabled_in_broker = true
      }
    }
  }

  tags = var.tags
}