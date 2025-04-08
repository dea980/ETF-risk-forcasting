# sagemaker/register_model.py
import boto3

s3_model_path = "s3://my-bucket/models/xgboost-model.tar.gz"
image_uri = "123456789.dkr.ecr.us-east-1.amazonaws.com/my-etf-image:latest"

client = boto3.client("sagemaker")

response = client.create_model(
    ModelName="ETF-XGBoost-Model",
    ExecutionRoleArn="arn:aws:iam::1234:role/SageMakerRole",
    PrimaryContainer={
        "Image": image_uri,
        "ModelDataUrl": s3_model_path
    }
)
