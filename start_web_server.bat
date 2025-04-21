@echo off
echo Starting ID Card Scanner Web Server with explicit credentials...
echo.

set GOOGLE_APPLICATION_CREDENTIALS=%~dp0n8n-test-456921-2c4224bba16d.json
echo Set Google Cloud credentials to: %GOOGLE_APPLICATION_CREDENTIALS%
echo.

python show_ip.py

echo Press Ctrl+C to stop the server
echo.

python web_server.py 