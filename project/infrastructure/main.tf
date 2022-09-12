terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  region  = var.aws_region
}

data "aws_caller_identity" "current_identity" {}

locals {
  account_id = data.aws_caller_identity.current_identity.account_id
}

module "s3_bucket" {
  source = "./modules/ec2"
  bucket_name = "${var.model_bucket}"
}

module "app_server" {
  source = "./modules/ec2"
  instance_name = "${modules.ec2.var.instance_name}"
}

output "model_bucket" {
  value = module.s3_bucket.name
}
