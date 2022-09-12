variable "instance_name" {
  description = "Value of the Name tag for the EC2 instance"
  type = string
  default = "MLOpsZoomcampProject"
}

variable "aws_region" {
  description = "AWS region to create resources"
  default     = "us-east-1"
}

variable "project_id" {
  description = "project_id"
  default = "mlops-zoomcamp"
}

variable "model_bucket" {
  description = "s3_bucket"
  default = "s3://mlflow-models-mlopszoomcamp"
}
