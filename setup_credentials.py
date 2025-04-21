import os
import sys
import json
import platform

def setup_credentials():
    print("Google Cloud Vision API Credentials Setup")
    print("=========================================")
    print("This script will help you set up your Google Cloud Vision credentials.")
    print("You'll need to have a Google Cloud service account key file (JSON) ready.")
    print()
    
    credentials_path = input("Enter the full path to your service account key file: ")
    
    # Check if file exists
    if not os.path.exists(credentials_path):
        print(f"Error: File not found at {credentials_path}")
        return False
    
    # Validate JSON format
    try:
        with open(credentials_path, 'r') as f:
            json.load(f)
    except json.JSONDecodeError:
        print("Error: The file is not valid JSON")
        return False
    
    # Set the environment variable
    if platform.system() == "Windows":
        # For Windows, we'll create a batch file to set the environment variable
        with open("set_credentials.bat", "w") as f:
            f.write(f"@echo off\n")
            f.write(f'set GOOGLE_APPLICATION_CREDENTIALS={credentials_path}\n')
            f.write(f'echo Google Cloud credentials set to: %GOOGLE_APPLICATION_CREDENTIALS%\n')
            f.write(f'python id_processor.py\n')
        
        print("\nCreated set_credentials.bat file.")
        print("Run this batch file before using the ID Processor:")
        print("    set_credentials.bat")
    else:
        # For Linux/Mac, we'll add the export command to a shell script
        with open("set_credentials.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f'export GOOGLE_APPLICATION_CREDENTIALS="{credentials_path}"\n')
            f.write('echo "Google Cloud credentials set to: $GOOGLE_APPLICATION_CREDENTIALS"\n')
            f.write('python id_processor.py\n')
        
        # Make the script executable
        os.chmod("set_credentials.sh", 0o755)
        
        print("\nCreated set_credentials.sh file.")
        print("Run this shell script before using the ID Processor:")
        print("    ./set_credentials.sh")
    
    print("\nYou can now run the ID Processor with:")
    if platform.system() == "Windows":
        print("    set_credentials.bat")
    else:
        print("    ./set_credentials.sh")
    
    return True

if __name__ == "__main__":
    setup_credentials() 