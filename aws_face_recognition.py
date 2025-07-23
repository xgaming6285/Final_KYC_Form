"""
AWS Rekognition Face Recognition Module
Replaces the face_recognition library with AWS Rekognition for face comparison
"""

import boto3
import cv2
import numpy as np
from PIL import Image
import io
import os
import base64

class AWSFaceRecognition:
    def __init__(self):
        """Initialize AWS Rekognition client."""
        try:
            # Initialize AWS Rekognition client
            self.rekognition = boto3.client('rekognition')
            print("AWS Rekognition client initialized successfully")
        except Exception as e:
            print(f"Error initializing AWS Rekognition: {str(e)}")
            self.rekognition = None
    
    def _image_to_bytes(self, image):
        """Convert different image formats to bytes for AWS Rekognition."""
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
            pil_image.save(img_byte_arr, format='JPEG')
            return img_byte_arr.getvalue()
        elif isinstance(image, Image.Image):
            # If it's a PIL Image
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            return img_byte_arr.getvalue()
        else:
            raise ValueError("Unsupported image format")
    
    def detect_faces(self, image):
        """Detect faces in an image using AWS Rekognition."""
        if not self.rekognition:
            raise Exception("AWS Rekognition client not initialized")
        
        try:
            image_bytes = self._image_to_bytes(image)
            
            response = self.rekognition.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['DEFAULT']
            )
            
            return response.get('FaceDetails', [])
        except Exception as e:
            print(f"Error detecting faces: {str(e)}")
            return []
    
    def compare_faces(self, source_image, target_image, similarity_threshold=80):
        """
        Compare two faces using AWS Rekognition.
        Returns similarity score and whether faces match.
        """
        if not self.rekognition:
            raise Exception("AWS Rekognition client not initialized")
        
        try:
            source_bytes = self._image_to_bytes(source_image)
            target_bytes = self._image_to_bytes(target_image)
            
            response = self.rekognition.compare_faces(
                SourceImage={'Bytes': source_bytes},
                TargetImage={'Bytes': target_bytes},
                SimilarityThreshold=similarity_threshold
            )
            
            face_matches = response.get('FaceMatches', [])
            
            if face_matches:
                # Get the highest similarity score
                best_match = max(face_matches, key=lambda x: x['Similarity'])
                similarity = best_match['Similarity']
                confidence = best_match['Face']['Confidence']
                
                return {
                    'success': True,
                    'similarity': similarity,
                    'confidence': confidence,
                    'match': similarity >= similarity_threshold,
                    'face_distance': (100 - similarity) / 100  # Convert to distance format for compatibility
                }
            else:
                return {
                    'success': False,
                    'similarity': 0,
                    'confidence': 0,
                    'match': False,
                    'face_distance': 1.0,  # Maximum distance (no match)
                    'message': 'No matching faces found'
                }
                
        except Exception as e:
            print(f"Error comparing faces: {str(e)}")
            return {
                'success': False,
                'similarity': 0,
                'confidence': 0,
                'match': False,
                'face_distance': 1.0,
                'error': str(e)
            }
    
    def compare_faces_s3(self, source_s3_bucket, source_s3_key, target_image, similarity_threshold=80):
        """
        Compare two faces using AWS Rekognition with S3 source image.
        This avoids the need to download the S3 image (no s3:GetObject permission needed).
        
        Args:
            source_s3_bucket: S3 bucket name containing the source image
            source_s3_key: S3 key (path) of the source image
            target_image: Target image (bytes, numpy array, or PIL Image)
            similarity_threshold: Minimum similarity threshold for a match
            
        Returns:
            dict: Comparison result with similarity, confidence, and match status
        """
        if not self.rekognition:
            raise Exception("AWS Rekognition client not initialized")
        
        try:
            target_bytes = self._image_to_bytes(target_image)
            
            response = self.rekognition.compare_faces(
                SourceImage={
                    'S3Object': {
                        'Bucket': source_s3_bucket,
                        'Name': source_s3_key
                    }
                },
                TargetImage={'Bytes': target_bytes},
                SimilarityThreshold=similarity_threshold
            )
            
            face_matches = response.get('FaceMatches', [])
            
            if face_matches:
                # Get the highest similarity score
                best_match = max(face_matches, key=lambda x: x['Similarity'])
                similarity = best_match['Similarity']
                confidence = best_match['Face']['Confidence']
                
                return {
                    'success': True,
                    'similarity': similarity,
                    'confidence': confidence,
                    'match': similarity >= similarity_threshold,
                    'face_distance': (100 - similarity) / 100  # Convert to distance format for compatibility
                }
            else:
                return {
                    'success': False,
                    'similarity': 0,
                    'confidence': 0,
                    'match': False,
                    'face_distance': 1.0,  # Maximum distance (no match)
                    'message': 'No matching faces found'
                }
                
        except Exception as e:
            print(f"Error comparing faces with S3 source: {str(e)}")
            return {
                'success': False,
                'similarity': 0,
                'confidence': 0,
                'match': False,
                'face_distance': 1.0,
                'error': str(e)
            }
    
    def compare_faces_s3_hybrid(self, source_s3_bucket, source_s3_key, target_image, similarity_threshold=80):
        """
        Hybrid AWS Rekognition face comparison that tries S3-based comparison first,
        then falls back to download-based comparison if service permissions aren't configured.
        
        Args:
            source_s3_bucket: S3 bucket name containing the source image
            source_s3_key: S3 key (path) of the source image
            target_image: Target image (bytes, numpy array, or PIL Image)
            similarity_threshold: Minimum similarity threshold for a match
            
        Returns:
            dict: Comparison result with similarity, confidence, and match status
        """
        print(f"Attempting S3-based face comparison: {source_s3_bucket}/{source_s3_key}")
        
        # First, try S3-based comparison (optimal if service permissions are configured)
        s3_based_failed = False
        s3_error_message = ""
        
        try:
            result = self.compare_faces_s3(source_s3_bucket, source_s3_key, target_image, similarity_threshold)
            
            # Check if the result indicates success or a legitimate failure (not permission-related)
            if result['success']:
                print("✓ S3-based comparison successful")
                return result
            elif 'error' in result:
                error_str = result['error']
                # Check for S3/permission-related errors that should trigger fallback
                permission_errors = [
                    'InvalidS3ObjectException',
                    'AccessDenied', 
                    'NoSuchKey',
                    'Forbidden',
                    'does not have permission',
                    'InvalidS3Object'
                ]
                
                if any(perm_error in error_str for perm_error in permission_errors):
                    print(f"S3-based comparison failed due to permissions/access: {error_str}")
                    s3_based_failed = True
                    s3_error_message = error_str
                else:
                    # Non-permission error, return the result as-is
                    print(f"S3-based comparison failed with non-permission error: {error_str}")
                    return result
            else:
                # No faces found or other legitimate failure
                print("S3-based comparison returned no matching faces")
                return result
                
        except Exception as e:
            error_str = str(e)
            print(f"S3-based comparison threw exception: {error_str}")
            
            # Check if this is a permission/access related exception
            permission_errors = [
                'InvalidS3ObjectException',
                'AccessDenied', 
                'NoSuchKey',
                'Forbidden',
                'does not have permission',
                'InvalidS3Object',
                'ClientError'
            ]
            
            if any(perm_error in error_str for perm_error in permission_errors):
                s3_based_failed = True
                s3_error_message = error_str
            else:
                # Non-permission exception, return error
                return {
                    'success': False,
                    'similarity': 0,
                    'confidence': 0,
                    'match': False,
                    'face_distance': 1.0,
                    'error': error_str
                }
        
        # If we reach here, S3-based comparison failed due to permissions, fall back to download
        if s3_based_failed:
            print(f"S3-based comparison failed due to permissions ({s3_error_message}), falling back to download...")
            
            try:
                # Download the source image using boto3 S3 client
                import boto3
                s3_client = boto3.client('s3')
                
                print(f"Downloading image from S3: {source_s3_bucket}/{source_s3_key}")
                response = s3_client.get_object(Bucket=source_s3_bucket, Key=source_s3_key)
                source_image_bytes = response['Body'].read()
                
                # Convert bytes to image format compatible with _image_to_bytes
                import io
                from PIL import Image
                source_image_pil = Image.open(io.BytesIO(source_image_bytes))
                
                print("✓ Successfully downloaded image, performing byte-based comparison")
                
                # Now use regular comparison with downloaded image
                return self.compare_faces(source_image_pil, target_image, similarity_threshold)
                
            except Exception as download_error:
                print(f"Download fallback also failed: {str(download_error)}")
                return {
                    'success': False,
                    'similarity': 0,
                    'confidence': 0,
                    'match': False,
                    'face_distance': 1.0,
                    'error': f"Both S3-based and download-based comparison failed. S3 error: {s3_error_message}, Download error: {str(download_error)}"
                }
        
        # This should not be reached, but just in case
        return {
            'success': False,
            'similarity': 0,
            'confidence': 0,
            'match': False,
            'face_distance': 1.0,
            'error': "Unexpected error in S3 hybrid comparison"
        }
    
    def extract_face_region(self, image, face_details=None):
        """
        Extract face region from image using AWS Rekognition face detection.
        Returns the cropped face image.
        """
        if face_details is None:
            face_details = self.detect_faces(image)
        
        if not face_details:
            return None
        
        # Convert image to numpy array if needed
        if isinstance(image, str):
            img_array = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img_array = np.array(image)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_array = image
        
        # Get the first (most confident) face
        face = face_details[0]
        bbox = face['BoundingBox']
        
        # Convert relative coordinates to absolute
        height, width = img_array.shape[:2]
        left = int(bbox['Left'] * width)
        top = int(bbox['Top'] * height)
        face_width = int(bbox['Width'] * width)
        face_height = int(bbox['Height'] * height)
        
        # Extract face region
        face_region = img_array[top:top+face_height, left:left+face_width]
        
        return face_region
    
    def face_locations(self, image):
        """
        Get face locations in format compatible with face_recognition library.
        Returns list of (top, right, bottom, left) tuples.
        """
        face_details = self.detect_faces(image)
        locations = []
        
        if isinstance(image, str):
            img_array = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        height, width = img_array.shape[:2]
        
        for face in face_details:
            bbox = face['BoundingBox']
            left = int(bbox['Left'] * width)
            top = int(bbox['Top'] * height)
            face_width = int(bbox['Width'] * width)
            face_height = int(bbox['Height'] * height)
            right = left + face_width
            bottom = top + face_height
            
            # Return in (top, right, bottom, left) format for compatibility
            locations.append((top, right, bottom, left))
        
        return locations

# Global instance
aws_face_recognition = AWSFaceRecognition()

# Compatibility functions to replace face_recognition library calls
def face_locations(image, model="hog"):
    """Compatibility function for face_recognition.face_locations()"""
    return aws_face_recognition.face_locations(image)

def face_encodings(image, known_face_locations=None):
    """
    Compatibility function for face_recognition.face_encodings()
    Note: AWS Rekognition doesn't use encodings, so this returns a placeholder
    """
    locations = known_face_locations or face_locations(image)
    # Return placeholder encodings (not used in AWS comparison)
    return [f"aws_face_{i}" for i in range(len(locations))]

def face_distance(known_face_encodings, face_encoding_to_check):
    """
    Compatibility function for face_recognition.face_distance()
    Note: This is a placeholder since AWS Rekognition handles comparison differently
    """
    # Return placeholder distance (will be replaced by direct AWS comparison)
    return [0.5] * len(known_face_encodings)

def compare_faces_aws(source_image, target_image, similarity_threshold=80):
    """
    Direct AWS Rekognition face comparison function.
    Use this instead of the compatibility functions for better results.
    """
    return aws_face_recognition.compare_faces(source_image, target_image, similarity_threshold)

def compare_faces_aws_s3(source_s3_bucket, source_s3_key, target_image, similarity_threshold=80):
    """
    Direct AWS Rekognition face comparison function using S3 source image.
    Falls back to download-based comparison if S3 service permissions aren't configured.
    
    Args:
        source_s3_bucket: S3 bucket name containing the source image
        source_s3_key: S3 key (path) of the source image  
        target_image: Target image (bytes, numpy array, or PIL Image)
        similarity_threshold: Minimum similarity threshold for a match
        
    Returns:
        dict: Comparison result with similarity, confidence, and match status
    """
    return aws_face_recognition.compare_faces_s3_hybrid(source_s3_bucket, source_s3_key, target_image, similarity_threshold) 