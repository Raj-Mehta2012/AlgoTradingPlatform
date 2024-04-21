import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
import os
import json
import boto3
from dotenv import load_dotenv

def fetch_data():
    start_date = datetime(2020, 1, 1)
    end_date = date.today()
    data = yf.download('META', start=start_date, end=end_date)
    data['Typical_Price'] = data[['High', 'Low', 'Close']].mean(axis=1)
    data['lrets'] = (np.log(data['Close']) - np.log(data['Close'].shift(1))) * 100
    data.dropna(subset=['lrets'], inplace=True)
    print("Data fetched from Yahoo Finance.")
    data.to_csv("METAData.csv", encoding='utf-8', index=False)

load_dotenv()
fetch_data()

# Access AWS credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Specify your S3 bucket and path
s3_bucket = 'algo-trading-script'
script_path = f's3://{s3_bucket}/algotradingstrategy.py'

# Get the SageMaker execution role
role = 'arn:aws:iam::149023223962:role/service-role/AmazonSageMaker-ExecutionRole-20240404T184210'
script_path = f's3://{s3_bucket}/code_and_req.tar.gz'
sklearn_version = '0.23-1'

# Create an S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Specify the bucket name and file path
bucket_name = 'algo-trading-script'
file_path = 'METAData.csv'

# Upload the file to S3
try:
    s3_client.upload_file(file_path, bucket_name, file_path)
    print(f'File {file_path} uploaded successfully to S3 bucket {bucket_name}')
except Exception as e:
    print(f'Error uploading file to S3: {e}')

estimator = SKLearn(
    entry_point='algotradingstrategy.py',  # Entry point for training
    source_dir='s3://algo-trading-script/code_and_req.tar.gz'.format(s3_bucket),  # Path to the compressed source code
    role=role,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    py_version='py3',
    sagemaker_session=sagemaker.Session()
)

# Assuming the training data is directly accessible and not compressed
estimator.fit({'train': f's3://{s3_bucket}/METAData.csv'})

# Deploy the model to an endpoint
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)