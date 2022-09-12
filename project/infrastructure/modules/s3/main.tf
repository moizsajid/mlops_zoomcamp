resource "aws_s3_bucket" "s3_bucket" {
  bucket = var.bucket_name
  acl    = "public"
  force_destroy = false
}

output "name" {
  value = aws_s3_bucket.s3_bucket.bucket
}
