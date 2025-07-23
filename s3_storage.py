"""
S3 Storage Utility Module
Handles uploading and downloading of images to/from AWS S3 bucket for KYC processing
"""

import boto3
import os
import io
from PIL import Image
import cv2
import numpy as np
from botocore.exceptions import ClientError, NoCredentialsError
import uuid
from datetime import datetime

class S3Storage:
    def __init__(self, bucket_name="kyc-form-uploads"):
        """Initialize S3 client and set bucket name."""
        self.bucket_name = bucket_name
        try:
            self.s3_client = boto3.client('s3')
            print(f"S3 client initialized successfully for bucket: {bucket_name}")
            # Test bucket access
            self._test_bucket_access()
        except NoCredentialsError:
            print("Error: AWS credentials not found. Please configure your AWS credentials.")
            self.s3_client = None
        except Exception as e:
            print(f"Error initializing S3 client: {str(e)}")
            self.s3_client = None
    
    def _test_bucket_access(self):
        """Test if we can access the S3 bucket."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✓ Successfully connected to S3 bucket: {self.bucket_name}")
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                print(f"Warning: Bucket {self.bucket_name} does not exist. It will be created on first upload.")
            else:
                print(f"Error accessing bucket {self.bucket_name}: {str(e)}")
        except Exception as e:
            print(f"Error testing bucket access: {str(e)}")
    
    def _image_to_bytes(self, image, format='JPEG', quality=90):
        """Convert different image formats to bytes for S3 upload."""
        if isinstance(image, str):
            # If it's a file path
            with open(image, 'rb') as img_file:
                return img_file.read()
        elif isinstance(image, np.ndarray):
            # If it's a numpy array (OpenCV format)
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format=format, quality=quality)
            return img_byte_arr.getvalue()
        elif isinstance(image, Image.Image):
            # If it's a PIL Image
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=format, quality=quality)
            return img_byte_arr.getvalue()
        else:
            raise ValueError("Unsupported image format")
    
    def upload_image(self, image, session_id, image_type, file_extension='jpg'):
        """
        Upload an image to S3.
        
        Args:
            image: Image data (file path, numpy array, or PIL Image)
            session_id: Unique session identifier
            image_type: Type of image ('front', 'back', 'face', etc.)
            file_extension: File extension (default 'jpg')
        
        Returns:
            dict: Result containing success status, s3_key, and s3_url
        """
        if not self.s3_client:
            return {
                'success': False,
                'error': 'S3 client not initialized',
                's3_key': None,
                's3_url': None
            }
        
        try:
            # Generate S3 key (path)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"kyc-images/{session_id}/{timestamp}_{image_type}.{file_extension}"
            
            # Convert image to bytes
            image_bytes = self._image_to_bytes(image)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=image_bytes,
                ContentType=f'image/{file_extension}',
                Metadata={
                    'session_id': session_id,
                    'image_type': image_type,
                    'upload_timestamp': timestamp
                }
            )
            
            # Generate S3 URL
            s3_url = f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"
            
            print(f"✓ Successfully uploaded {image_type} image to S3: {s3_key}")
            
            return {
                'success': True,
                'error': None,
                's3_key': s3_key,
                's3_url': s3_url
            }
            
        except Exception as e:
            print(f"Error uploading image to S3: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                's3_key': None,
                's3_url': None
            }
    
    def upload_video(self, video_data, session_id, camera_type, file_extension='webm'):
        """
        Upload a video to S3.
        
        Args:
            video_data: Video blob data
            session_id: Unique session identifier
            camera_type: Type of camera ('front_camera', 'back_camera')
            file_extension: Video file extension (default 'webm')
        
        Returns:
            dict: Result containing success status, s3_key, and s3_url
        """
        if not self.s3_client:
            return {
                'success': False,
                'error': 'S3 client not initialized',
                's3_key': None,
                's3_url': None
            }
        
        try:
            # Generate S3 key (path) for video
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"kyc-videos/{session_id}/video_session_{camera_type}_{timestamp}.{file_extension}"
            
            # Upload video to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=video_data,
                ContentType=f'video/{file_extension}',
                Metadata={
                    'session_id': session_id,
                    'camera_type': camera_type,
                    'upload_timestamp': timestamp,
                    'content_type': 'session_video'
                }
            )
            
            # Generate S3 URL
            s3_url = f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"
            
            print(f"✓ Successfully uploaded {camera_type} video to S3: {s3_key}")
            
            return {
                'success': True,
                'error': None,
                's3_key': s3_key,
                's3_url': s3_url
            }
            
        except Exception as e:
            print(f"Error uploading video to S3: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                's3_key': None,
                's3_url': None
            }

    def download_image(self, s3_key):
        """
        Download an image from S3.
        
        Args:
            s3_key: S3 key (path) of the image
            
        Returns:
            bytes: Image data if successful, None if failed
        """
        if not self.s3_client:
            print("Error: S3 client not initialized")
            return None
            
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response['Body'].read()
        except Exception as e:
            print(f"Error downloading image from S3: {str(e)}")
            return None
    
    def get_image_url(self, s3_key, expiration=3600):
        """
        Generate a presigned URL for an S3 object.
        
        Args:
            s3_key: S3 key (path) of the image
            expiration: URL expiration time in seconds (default 1 hour)
            
        Returns:
            str: Presigned URL if successful, None if failed
        """
        if not self.s3_client:
            print("Error: S3 client not initialized")
            return None
            
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            print(f"Error generating presigned URL: {str(e)}")
            return None
    
    def get_video_url(self, s3_key, expiration=3600):
        """
        Generate a presigned URL for a video S3 object.
        
        Args:
            s3_key: S3 key (path) of the video
            expiration: URL expiration time in seconds (default 1 hour)
            
        Returns:
            str: Presigned URL if successful, None if failed
        """
        if not self.s3_client:
            print("Error: S3 client not initialized")
            return None
            
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            print(f"Error generating video presigned URL: {str(e)}")
            return None

    def list_session_images(self, session_id):
        """
        List all images for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            list: List of S3 keys for the session
        """
        if not self.s3_client:
            print("Error: S3 client not initialized")
            return []
            
        try:
            prefix = f"kyc-images/{session_id}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            keys = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    keys.append(obj['Key'])
            
            return keys
        except Exception as e:
            print(f"Error listing session images: {str(e)}")
            return []
    
    def delete_image(self, s3_key):
        """
        Delete an image from S3.
        
        Args:
            s3_key: S3 key (path) of the image to delete
            
        Returns:
            bool: True if successful, False if failed
        """
        if not self.s3_client:
            print("Error: S3 client not initialized")
            return False
            
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            print(f"✓ Successfully deleted image from S3: {s3_key}")
            return True
        except Exception as e:
            print(f"Error deleting image from S3: {str(e)}")
            return False

# Global instance
s3_storage = S3Storage()

# Convenience functions
def upload_image_to_s3(image, session_id, image_type, file_extension='jpg'):
    """Upload an image to S3 (convenience function)."""
    return s3_storage.upload_image(image, session_id, image_type, file_extension)

def get_image_from_s3(s3_key):
    """Download an image from S3 (convenience function)."""
    return s3_storage.download_image(s3_key)

def get_s3_image_url(s3_key, expiration=3600):
    """Get a presigned URL for an S3 image (convenience function)."""
    return s3_storage.get_image_url(s3_key, expiration) 

def upload_video_to_s3(video_data, session_id, camera_type, file_extension='webm'):
    """
    Convenience function to upload video using the global s3_storage instance.
    
    Args:
        video_data: Video blob data
        session_id: Unique session identifier
        camera_type: Type of camera ('front_camera', 'back_camera')
        file_extension: Video file extension (default 'webm')
    
    Returns:
        dict: Result containing success status, s3_key, and s3_url
    """
    return s3_storage.upload_video(video_data, session_id, camera_type, file_extension) 