@echo off
echo Starting ID Card Scanner Web Server...
echo.

python show_ip.py

echo Press Ctrl+C to stop the server
echo.

python web_server.py 