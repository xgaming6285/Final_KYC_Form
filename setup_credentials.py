#!/usr/bin/env python3
"""
AWS Credentials Setup Script
Sets up AWS credentials for the KYC application to use AWS Rekognition
"""

import os
import json

def setup_aws_credentials():
    """Set up AWS credentials for the application."""
    
    # AWS credentials

    
    # Set environment variables
    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
    os.environ['AWS_DEFAULT_REGION'] = aws_region
    
    # Create credentials file for boto3
    aws_dir = os.path.expanduser('~/.aws')
    os.makedirs(aws_dir, exist_ok=True)
    
    # Write credentials file
    credentials_content = f"""[default]
aws_access_key_id = {aws_access_key_id}
aws_secret_access_key = {aws_secret_access_key}
"""
    
    with open(os.path.join(aws_dir, 'credentials'), 'w') as f:
        f.write(credentials_content)
    
    # Write config file
    config_content = f"""[default]
region = {aws_region}
output = json
"""
    
    with open(os.path.join(aws_dir, 'config'), 'w') as f:
        f.write(config_content)
    
    print("AWS credentials have been set up successfully!")
    print(f"Region: {aws_region}")
    print(f"Access Key ID: {aws_access_key_id[:10]}...")
    
    return True

if __name__ == "__main__":
    setup_aws_credentials() 