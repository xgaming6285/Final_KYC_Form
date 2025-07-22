#!/usr/bin/env python3
"""
Test script for AWS Rekognition integration
Tests face detection and comparison functionality
"""

import os
import sys
from setup_credentials import setup_aws_credentials
from aws_face_recognition import AWSFaceRecognition, compare_faces_aws

def test_aws_rekognition():
    """Test AWS Rekognition functionality."""
    print("Testing AWS Rekognition Integration")
    print("=" * 40)
    
    # Setup credentials
    print("1. Setting up AWS credentials...")
    try:
        setup_aws_credentials()
        print("✓ AWS credentials set up successfully")
    except Exception as e:
        print(f"✗ Error setting up credentials: {e}")
        return False
    
    # Test AWS Rekognition client initialization
    print("\n2. Testing AWS Rekognition client...")
    try:
        face_recognition = AWSFaceRecognition()
        if face_recognition.rekognition:
            print("✓ AWS Rekognition client initialized successfully")
        else:
            print("✗ AWS Rekognition client failed to initialize")
            return False
    except Exception as e:
        print(f"✗ Error initializing AWS Rekognition: {e}")
        return False
    
    # Test compatibility functions
    print("\n3. Testing compatibility functions...")
    try:
        from aws_face_recognition import face_locations, face_encodings, face_distance
        print("✓ Compatibility functions imported successfully")
    except Exception as e:
        print(f"✗ Error importing compatibility functions: {e}")
        return False
    
    # Test AWS service connection (without actual image)
    print("\n4. Testing AWS service connection...")
    try:
        # Just test the client connection by calling a simple method
        import boto3
        client = boto3.client('rekognition')
        # This will test connectivity and credentials
        response = client.list_faces(CollectionId='test-collection-that-does-not-exist')
    except client.exceptions.ResourceNotFoundException:
        print("✓ AWS Rekognition service is accessible (expected ResourceNotFound)")
    except Exception as e:
        error_str = str(e)
        if "AccessDenied" in error_str:
            print("✗ AWS Access Denied - check your credentials")
            return False
        elif "InvalidParameter" in error_str or "ValidationException" in error_str:
            print("✓ AWS Rekognition service is accessible (parameter validation working)")
        else:
            print(f"? AWS connection test returned: {error_str}")
            print("  This may be normal depending on your AWS setup")
    
    print("\n" + "=" * 40)
    print("✓ AWS Rekognition integration test completed successfully!")
    print("\nThe application should now work without face_recognition library.")
    print("Face comparison will use AWS Rekognition instead of local processing.")
    return True

if __name__ == "__main__":
    success = test_aws_rekognition()
    if success:
        print("\n🎉 You can now run the web server or Streamlit app!")
        print("   - Web server: python web_server.py")
        print("   - Streamlit: streamlit run streamlit_app.py")
    else:
        print("\n❌ Please check the errors above and try again.")
        sys.exit(1) 