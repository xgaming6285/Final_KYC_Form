@echo off
echo Starting KYC Web Server with AWS Rekognition...
echo.

echo Setting up AWS credentials...
python setup_credentials.py
echo.

python show_ip.py

echo Press Ctrl+C to stop the server
echo.

python web_server.py 