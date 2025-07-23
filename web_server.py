from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, current_app
import os
import uuid
import cv2
import numpy as np
from PIL import Image
import io
import base64
import sys
import json
# Google Cloud Vision imports disabled - no longer extracting text
# from google.cloud import vision
# from google.oauth2 import service_account
import datetime
from aws_face_recognition import face_locations, face_encodings, face_distance, compare_faces_aws, compare_faces_aws_s3
from image_quality_validator import ImageQualityValidator
from s3_storage import upload_image_to_s3, get_s3_image_url, s3_storage, upload_video_to_s3

# Google Cloud credentials setup disabled
# credential_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'n8n-test-456921-2c4224bba16d.json')
# if os.path.exists(credential_path):
#     os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
#     print(f"Set Google Cloud credentials to: {credential_path}")
# else:
#     print(f"Warning: Credentials file not found at {credential_path}")
#     print("Scanning features that require Google Cloud Vision may not work.")
print("Google Cloud Vision disabled - only image capture functionality available")

# Initialize image quality validator instead of text extraction
quality_validator = ImageQualityValidator()
print("Initialized Image Quality Validator for ID validation")

# Simple in-memory session storage to track uploaded S3 keys
# This avoids the need for s3:ListBucket permission
session_storage = {}

# Add temporary local storage tracking
local_temp_storage = {}

# Video recording session storage
video_sessions = {}

def store_session_image(session_id, image_type, s3_key):
    """Store S3 key for a session image."""
    if session_id not in session_storage:
        session_storage[session_id] = {}
    session_storage[session_id][image_type] = s3_key
    print(f"Stored {image_type} S3 key for session {session_id}: {s3_key}")

def store_local_temp_image(session_id, image_type, local_path):
    """Store local temporary image path for a session."""
    if session_id not in local_temp_storage:
        local_temp_storage[session_id] = {}
    local_temp_storage[session_id][image_type] = local_path
    print(f"Stored {image_type} local temp path for session {session_id}: {local_path}")

def get_local_temp_image(session_id, image_type):
    """Get local temporary image path for a session."""
    if session_id in local_temp_storage and image_type in local_temp_storage[session_id]:
        return local_temp_storage[session_id][image_type]
    return None

def cleanup_local_temp_files(session_id):
    """Clean up temporary local files for a session."""
    if session_id in local_temp_storage:
        for image_type, local_path in local_temp_storage[session_id].items():
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
                    print(f"Cleaned up temporary file: {local_path}")
            except Exception as e:
                print(f"Error cleaning up temporary file {local_path}: {str(e)}")

def get_session_image(session_id, image_type):
    """Get S3 key for a session image."""
    if session_id in session_storage and image_type in session_storage[session_id]:
        return session_storage[session_id][image_type]
    return None

def get_all_session_images(session_id):
    """Get all S3 keys for a session."""
    if session_id in session_storage:
        return session_storage[session_id]
    return {}

def cleanup_session(session_id):
    """Remove session data from memory storage."""
    if session_id in session_storage:
        del session_storage[session_id]
        print(f"Cleaned up session storage for session {session_id}")
    
    # Also cleanup local temporary files
    cleanup_local_temp_files(session_id)

def cleanup_old_sessions():
    """Clean up old sessions to prevent memory buildup (could be called periodically)."""
    # For now, just log the number of active sessions
    print(f"Active sessions in memory: {len(session_storage)}")

def store_session_video(session_id, camera_type, s3_key):
    """Store S3 key for a session video."""
    if session_id not in video_sessions:
        video_sessions[session_id] = {}
    video_sessions[session_id][camera_type] = s3_key
    print(f"Stored {camera_type} video S3 key for session {session_id}: {s3_key}")

app = Flask(__name__, static_folder='static', static_url_path='/static')
# Use absolute path for uploads directory
upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = upload_dir
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
print(f"Upload directory set to: {app.config['UPLOAD_FOLDER']}")

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_extracted_data(document_type, side, extracted_data):
    """Save the extracted data to a JSON file with timestamp."""
    # Create a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"extracted_data_{timestamp}.json"
    
    # If file exists already, load its contents, otherwise initialize a new dictionary
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {
            "document_type": document_type,
            "front_side": {},
            "back_side": {}
        }
    
    # Remove document_type from extracted_data if it exists
    if "document_type" in extracted_data:
        del extracted_data["document_type"]
    
    # Update the data with new extracted information
    if side == "front":
        data["front_side"] = extracted_data
    else:
        data["back_side"] = extracted_data
    
    # Save the data
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved extracted data to {filename}")
    return filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/simple_capture')
def simple_capture():
    return render_template('simple_capture.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/process_id', methods=['POST'])
def process_id():
    if 'file' not in request.files and 'image_data' not in request.form:
        return jsonify({'error': 'No file or image data provided'}), 400
    
    result = {}
    document_type = request.form.get('document_type', 'bulgarian_id')
    side = request.form.get('side', 'front')
    
    # Generate a session ID to link front and back images
    session_id = request.form.get('session_id', str(uuid.uuid4()))
    
    # Process uploaded file
    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Read image data directly without saving locally
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'error': 'Could not read image file'}), 400
            
            # Upload image to S3
            s3_result = upload_image_to_s3(image, session_id, side)
            
            if not s3_result['success']:
                return jsonify({'error': f'Failed to upload image to S3: {s3_result["error"]}'}), 500
            
            # Store S3 key in session storage for later retrieval
            store_session_image(session_id, side, s3_result['s3_key'])
            
            # For front ID images, also save locally for AWS Rekognition face verification
            if side == 'front':
                try:
                    # Create a unique filename for the temporary front image
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_filename = f"temp_front_{session_id}_{timestamp}.jpg"
                    temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
                    
                    # Save the image locally
                    cv2.imwrite(temp_filepath, image)
                    
                    # Store the local path for later use in face verification
                    store_local_temp_image(session_id, 'front', temp_filepath)
                    print(f"Saved front ID image locally for face verification: {temp_filepath}")
                    
                except Exception as e:
                    print(f"Warning: Failed to save front image locally: {str(e)}")
                    # Continue execution - S3 upload was successful, local save is just for face verification
            
            # Skip quality validation - just save the image and proceed
            result = {
                'success': True,
                'image_path': s3_result['s3_url'],  # S3 URL instead of local path
                's3_key': s3_result['s3_key'],      # Store S3 key for later retrieval
                'session_id': session_id,
                'message': f'ID {side} side captured successfully!',
                'is_valid': True  # Always consider valid since user confirms manually
            }
            
            # Add next_step field to redirect to back side when front is complete
            if side == 'front':
                result['next_step'] = 'back'
            elif side == 'back':
                result['next_step'] = 'face'
    
    # Process image data from camera
    elif 'image_data' in request.form:
        image_data = request.form['image_data']
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64 image
            image_bytes = io.BytesIO(base64.b64decode(image_data))
            image_pil = Image.open(image_bytes)
            
            # Upload image to S3
            s3_result = upload_image_to_s3(image_pil, session_id, side)
            
            if not s3_result['success']:
                return jsonify({'error': f'Failed to upload image to S3: {s3_result["error"]}'}), 500
            
            print(f"Uploaded captured image to S3: {s3_result['s3_key']}")
            
            # Store S3 key in session storage for later retrieval
            store_session_image(session_id, side, s3_result['s3_key'])
            
            # For front ID images, also save locally for AWS Rekognition face verification
            if side == 'front':
                try:
                    # Create a unique filename for the temporary front image
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_filename = f"temp_front_{session_id}_{timestamp}.jpg"
                    temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
                    
                    # Save the PIL image locally
                    image_pil.save(temp_filepath, 'JPEG')
                    
                    # Store the local path for later use in face verification
                    store_local_temp_image(session_id, 'front', temp_filepath)
                    print(f"Saved front ID image locally for face verification: {temp_filepath}")
                    
                except Exception as e:
                    print(f"Warning: Failed to save front image locally: {str(e)}")
                    # Continue execution - S3 upload was successful, local save is just for face verification
            
            # Skip quality validation - just save the image and proceed
            result = {
                'success': True,
                'image_path': s3_result['s3_url'],  # S3 URL instead of local path
                's3_key': s3_result['s3_key'],      # Store S3 key for later retrieval
                'session_id': session_id,
                'message': f'ID {side} side captured successfully!',
                'is_valid': True  # Always consider valid since user confirms manually
            }
            
            # Add next_step field to redirect to back side when front is complete
            if side == 'front':
                result['next_step'] = 'back'
            elif side == 'back':
                result['next_step'] = 'face'
                
        except Exception as e:
            return jsonify({'error': f'Error processing image data: {str(e)}'}), 400
    
    return jsonify(result)

@app.route('/get_validation_info', methods=['GET'])
def get_validation_info():
    """Endpoint to get validation information for a specific document type and side."""
    document_type = request.args.get('document_type', 'bulgarian_id')
    side = request.args.get('side', 'front')
    
    # Return validation criteria
    return jsonify({
        'document_type': document_type,
        'side': side,
        'validation_criteria': {
            'blur_detection': 'Image must be clear and not blurred',
            'corner_detection': 'All 4 corners of ID card must be visible',
            'lighting_check': 'Image must have adequate lighting (not too dark or bright)',
            'sharpness_check': 'Image must have sufficient detail and sharpness'
        },
        'requirements': [
            'Hold camera steady to avoid blur',
            'Ensure entire ID card is visible in frame',
            'Use good lighting, avoid shadows',
            'Make sure ID text is readable'
        ]
    })

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a video frame for real-time ID card detection and quality validation."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    document_type = request.form.get('document_type', 'bulgarian_id')
    side = request.form.get('side', 'front')
    session_id = request.form.get('session_id', str(uuid.uuid4()))
    
    try:
        # Get image file
        file = request.files['image']
        
        # Read image with OpenCV
        img_stream = file.read()
        img_array = np.frombuffer(img_stream, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Could not read image file'}), 400
        
        # Validate image quality and detect ID card
        validation_result = quality_validator.validate_id_image(image)
        feedback = quality_validator.get_quality_feedback(validation_result)
        
        # Check if corners were detected
        if not validation_result['corners_detected']:
            return jsonify({
                'cardDetected': False,
                'validation': validation_result,
                'feedback': feedback,
                'message': 'No ID card detected in frame. Position the entire ID card within the frame.',
                'quality_score': int(validation_result['quality_score'])
            })
        
        # Convert corners to a more usable format for front-end
        formatted_corners = []
        if validation_result['corners']:
            for corner in validation_result['corners']:
                formatted_corners.append({
                    'x': int(corner['x']),
                    'y': int(corner['y'])
                })
        
        # Return the validation results
        return jsonify({
            'cardDetected': True,
            'corners': formatted_corners,
            'validation': validation_result,
            'feedback': feedback,
            'quality_score': int(validation_result['quality_score']),
            'message': feedback['message'],
            'complete': validation_result['is_valid'],
            'next_step': 'back' if validation_result['is_valid'] and side == 'front' else ('face' if validation_result['is_valid'] and side == 'back' else None)
        })
    
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Error processing frame: {str(e)}',
            'cardDetected': False
        }), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Legacy route for backward compatibility - files are now stored in S3."""
    return jsonify({'error': 'Files are now stored in S3. Use the S3 URLs provided in the response.'}), 404

@app.route('/s3_image/<session_id>/<image_type>')
def get_s3_image(session_id, image_type):
    """Get a presigned URL for an S3 image."""
    try:
        # Get the S3 key from session storage (avoids s3:ListBucket permission)
        target_s3_key = get_session_image(session_id, image_type)
        
        if not target_s3_key:
            return jsonify({'error': 'Image not found'}), 404
        
        # Generate presigned URL
        presigned_url = get_s3_image_url(target_s3_key)
        if presigned_url:
            return redirect(presigned_url)
        else:
            return jsonify({'error': 'Failed to generate image URL'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Error retrieving image: {str(e)}'}), 500

@app.route('/face_verification')
def face_verification():
    """Render the face verification page."""
    return render_template('face_verification.html')

@app.route('/verify_face', methods=['POST'])
def verify_face():
    """Process face verification by comparing live face with ID card photo."""
    try:
        data = request.get_json()
        
        if not data or 'image_data' not in data or 'session_id' not in data:
            return jsonify({'success': False, 'message': 'Missing required data'}), 400
        
        # Get session ID and image data
        session_id = data['session_id']
        image_data = data['image_data']
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        face_image = Image.open(io.BytesIO(image_bytes))
        
        # Upload face image to S3
        s3_result = upload_image_to_s3(face_image, session_id, 'face')
        
        if not s3_result['success']:
            return jsonify({
                'success': False,
                'message': f'Failed to upload face image to S3: {s3_result["error"]}'
            }), 500
        
        # Store face image S3 key in session storage
        store_session_image(session_id, 'face', s3_result['s3_key'])
        
        # Convert to a format usable by face_recognition
        face_image_np = np.array(face_image)
        face_image_rgb = cv2.cvtColor(face_image_np, cv2.COLOR_BGR2RGB)
        
        # Try to get the local temporary front image first
        local_front_image_path = get_local_temp_image(session_id, 'front')
        
        if local_front_image_path and os.path.exists(local_front_image_path):
            print(f"Using local front ID image for face verification: {local_front_image_path}")
            
            # Use local image for AWS Rekognition comparison
            try:
                # Convert face image to format compatible with AWS Rekognition  
                face_image_pil = Image.fromarray(cv2.cvtColor(face_image_np, cv2.COLOR_BGR2RGB))
                
                # Use AWS Rekognition with local front image
                comparison_result = compare_faces_aws(
                    source_image=local_front_image_path,  # Local front ID image
                    target_image=face_image_pil,  # Captured face
                    similarity_threshold=85  # Higher threshold for better security
                )
                
                # Generate S3 URL for display purposes (if available)
                id_front_s3_key = get_session_image(session_id, 'front')
                id_photo_url = get_s3_image_url(id_front_s3_key) if id_front_s3_key else None
                
            except Exception as e:
                print(f"Error in local AWS Rekognition face comparison: {str(e)}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'message': f'Face verification could not be completed using local image. AWS Rekognition error: {str(e)}',
                    'face_distance': 1.0
                })
        
        else:
            # Fallback to S3-based comparison if local image is not available
            print("Local front image not found, falling back to S3-based comparison")
            
            # Get the ID card front image S3 key from session storage
            id_front_s3_key = get_session_image(session_id, 'front')
            
            if not id_front_s3_key:
                return jsonify({
                    'success': False,
                    'message': 'ID card front image not found. Please complete the ID card scanning process first.'
                }), 400
            
            # For debugging and display purposes
            print(f"Using ID image from S3: {id_front_s3_key}")
            
            # Generate presigned URL for display purposes
            id_photo_url = get_s3_image_url(id_front_s3_key)
            
            # Use AWS Rekognition to compare faces directly with S3 reference
            try:
                # Convert face image to format compatible with AWS Rekognition  
                face_image_pil = Image.fromarray(cv2.cvtColor(face_image_np, cv2.COLOR_BGR2RGB))
                
                # Use AWS Rekognition S3-based comparison (no download required)
                comparison_result = compare_faces_aws_s3(
                    source_s3_bucket=s3_storage.bucket_name,  # S3 bucket name
                    source_s3_key=id_front_s3_key,  # S3 key for ID card image
                    target_image=face_image_pil,  # Captured face
                    similarity_threshold=85  # Higher threshold for better security
                )
            except Exception as e:
                print(f"Error in S3-based AWS Rekognition face comparison: {str(e)}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'message': f'Face verification could not be completed. AWS Rekognition S3 error: {str(e)}',
                    'face_distance': 1.0,
                    'id_photo_url': id_photo_url
                })
            
        print(f"AWS Rekognition comparison result: {comparison_result}")
        
        # Cleanup local temporary files after face verification
        try:
            cleanup_local_temp_files(session_id)
        except Exception as cleanup_error:
            print(f"Warning: Error during cleanup: {str(cleanup_error)}")
        
        if comparison_result['success']:
            similarity = comparison_result['similarity']
            confidence = comparison_result['confidence']
            is_match = comparison_result['match']
            face_distance = comparison_result['face_distance']
            
            # Use the full ID card image URL for display
            id_display_url = id_photo_url if 'id_photo_url' in locals() else None
            
            return jsonify({
                'success': is_match,
                'message': f'Face verification {"successful" if is_match else "failed"}. Similarity: {similarity:.1f}%, Confidence: {confidence:.1f}%' if is_match else 
                           f'Face verification failed. The captured face does not match the ID card photo. Similarity: {similarity:.1f}%',
                'face_distance': float(face_distance),
                'similarity': float(similarity),
                'confidence': float(confidence)
                # Removed id_photo_url to prevent comparison display
            })
        else:
            # AWS Rekognition failed to find matching faces
            error_msg = comparison_result.get('message', 'No matching faces found')
            if 'error' in comparison_result:
                error_msg = f"AWS Rekognition error: {comparison_result['error']}"
            
            return jsonify({
                'success': False,
                'message': f'Face verification failed. {error_msg}',
                'face_distance': 1.0,
                'similarity': 0.0,
                'confidence': 0.0
                # Removed id_photo_url to prevent comparison display
            })
    
    except Exception as e:
        print(f"Error in face verification: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error during face verification: {str(e)}'
        }), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Upload session video to S3."""
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    session_id = request.form.get('session_id')
    camera_type = request.form.get('camera_type', 'front_camera')  # front_camera or back_camera
    
    if not session_id:
        return jsonify({'success': False, 'error': 'Session ID is required'}), 400
    
    if not video_file:
        return jsonify({'success': False, 'error': 'No video data provided'}), 400
    
    try:
        # Read video data
        video_data = video_file.read()
        
        # Determine file extension from mimetype
        file_extension = 'mp4'  # Default to MP4
        if video_file.content_type:
            if 'mp4' in video_file.content_type:
                file_extension = 'mp4'
            elif 'webm' in video_file.content_type:
                file_extension = 'webm'
            elif 'avi' in video_file.content_type:
                file_extension = 'avi'
            elif 'mov' in video_file.content_type:
                file_extension = 'mov'
        
        # Upload video to S3
        upload_result = upload_video_to_s3(video_data, session_id, camera_type, file_extension)
        
        if upload_result['success']:
            # Store video S3 key in session storage
            store_session_video(session_id, camera_type, upload_result['s3_key'])
            
            return jsonify({
                'success': True,
                'message': f'{camera_type} video uploaded successfully',
                's3_key': upload_result['s3_key'],
                's3_url': upload_result['s3_url']
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to upload video: {upload_result["error"]}'
            }), 500
            
    except Exception as e:
        print(f"Error processing video upload: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/upload_backup_video_to_s3', methods=['POST'])
def upload_backup_video_to_s3():
    """Upload the most recent backup video to S3 as fallback."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Session ID is required'}), 400
        
        print(f"📤 Attempting fallback upload for session: {session_id}")
        
        # Find the most recent backup video file for this session
        upload_dir = os.path.join(current_app.root_path, 'uploads')
        if not os.path.exists(upload_dir):
            return jsonify({'success': False, 'error': 'Upload directory not found'}), 404
        
        # Look for backup files for this session
        backup_pattern = f"backup_{session_id}_*_backup_*.mp4"
        backup_files = []
        
        for filename in os.listdir(upload_dir):
            if filename.startswith(f"backup_{session_id}_") and filename.endswith('.mp4'):
                file_path = os.path.join(upload_dir, filename)
                file_stat = os.stat(file_path)
                backup_files.append({
                    'path': file_path,
                    'name': filename,
                    'size': file_stat.st_size,
                    'mtime': file_stat.st_mtime
                })
        
        if not backup_files:
            return jsonify({'success': False, 'error': 'No backup video files found for this session'}), 404
        
        # Sort by modification time (newest first) and file size (largest first)
        backup_files.sort(key=lambda x: (x['mtime'], x['size']), reverse=True)
        
        print(f"📁 Found {len(backup_files)} backup files for session {session_id}")
        
        # Upload all backup videos (up to 3 most recent/largest ones to avoid overwhelming S3)
        uploaded_videos = []
        failed_uploads = []
        max_uploads = min(3, len(backup_files))  # Limit to 3 videos max
        
        for i, backup_file in enumerate(backup_files[:max_uploads]):
            try:
                print(f"📤 Uploading backup {i+1}/{max_uploads}: {backup_file['name']} ({backup_file['size']} bytes)")
                
                # Read the video file
                with open(backup_file['path'], 'rb') as video_file:
                    video_data = video_file.read()
                
                if len(video_data) == 0:
                    print(f"⚠️ Skipping empty backup video: {backup_file['name']}")
                    failed_uploads.append(f"{backup_file['name']} (empty)")
                    continue
                
                # Upload to S3 with unique camera type
                camera_type = f'backup_{i+1}' if i > 0 else 'final_backup'
                upload_result = upload_video_to_s3(video_data, session_id, camera_type, 'mp4')
                
                if upload_result['success']:
                    # Store video S3 key in session storage
                    store_session_video(session_id, camera_type, upload_result['s3_key'])
                    
                    uploaded_videos.append({
                        'filename': backup_file['name'],
                        's3_key': upload_result['s3_key'],
                        's3_url': upload_result['s3_url'],
                        'size_mb': len(video_data) / 1024 / 1024
                    })
                    
                    print(f"✅ Backup video {i+1} uploaded to S3: {upload_result['s3_key']} ({len(video_data) / 1024 / 1024:.1f} MB)")
                else:
                    failed_uploads.append(f"{backup_file['name']} ({upload_result['error']})")
                    print(f"❌ Failed to upload backup {i+1}: {upload_result['error']}")
                    
            except Exception as e:
                failed_uploads.append(f"{backup_file['name']} (exception: {str(e)})")
                print(f"❌ Exception uploading backup {i+1}: {str(e)}")
        
        # Prepare response
        if uploaded_videos:
            total_size = sum(video['size_mb'] for video in uploaded_videos)
            success_message = f"{len(uploaded_videos)} videos uploaded ({total_size:.1f} MB total)"
            
            if failed_uploads:
                success_message += f", {len(failed_uploads)} failed"
            
            return jsonify({
                'success': True,
                'message': success_message,
                'uploaded_videos': uploaded_videos,
                'failed_uploads': failed_uploads if failed_uploads else None,
                'total_uploaded': len(uploaded_videos),
                'total_failed': len(failed_uploads)
            })
        else:
            return jsonify({
                'success': False,
                'error': f'All backup video uploads failed: {", ".join(failed_uploads)}'
            }), 500
            
    except Exception as e:
        print(f"Error uploading backup video: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/debug_log', methods=['POST'])
def debug_log():
    """Receive debug logs from client."""
    try:
        data = request.get_json()
        log_message = data.get('message', '')
        log_level = data.get('level', 'INFO')
        session_id = data.get('session_id', 'unknown')
        
        print(f"[CLIENT-{log_level}] [{session_id}] {log_message}")
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error receiving debug log: {str(e)}")
        return jsonify({'success': False}), 500

@app.route('/start_video_recording', methods=['POST'])
def start_video_recording():
    """Initialize video recording session."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Session ID is required'}), 400
        
        # Initialize video session tracking
        if session_id not in video_sessions:
            video_sessions[session_id] = {}
        
        print(f"Video recording session initialized for: {session_id}")
        
        return jsonify({
            'success': True,
            'message': 'Video recording session initialized',
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error initializing video recording: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/get_session_videos/<session_id>')
def get_session_videos(session_id):
    """Get all videos for a session."""
    try:
        session_videos = video_sessions.get(session_id, {})
        
        # Generate presigned URLs for videos
        video_urls = {}
        for camera_type, s3_key in session_videos.items():
            presigned_url = s3_storage.get_video_url(s3_key) if hasattr(s3_storage, 'get_video_url') else None
            video_urls[camera_type] = {
                's3_key': s3_key,
                'url': presigned_url
            }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'videos': video_urls
        })
        
    except Exception as e:
        print(f"Error retrieving session videos: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/save_video_backup', methods=['POST'])
def save_video_backup():
    """Save video backup to local storage and optionally S3."""
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    session_id = request.form.get('session_id')
    camera_type = request.form.get('camera_type', 'backup')
    is_backup = request.form.get('is_backup', 'false').lower() == 'true'
    
    if not session_id:
        return jsonify({'success': False, 'error': 'Session ID is required'}), 400
    
    if not video_file:
        return jsonify({'success': False, 'error': 'No video data provided'}), 400
    
    try:
        # Read video data
        video_data = video_file.read()
        
        # Determine file extension from mimetype
        file_extension = 'mp4'  # Default to MP4
        if video_file.content_type:
            if 'mp4' in video_file.content_type:
                file_extension = 'mp4'
            elif 'webm' in video_file.content_type:
                file_extension = 'webm'
        
        # Save locally first (always save backups locally)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        local_filename = f"backup_{session_id}_{camera_type}_{timestamp}.{file_extension}"
        local_path = os.path.join(app.config['UPLOAD_FOLDER'], local_filename)
        
        with open(local_path, 'wb') as f:
            f.write(video_data)
        
        print(f"💾 Video backup saved locally: {local_path} ({len(video_data) / 1024 / 1024:.2f} MB)")
        
        result = {
            'success': True,
            'local_path': local_path,
            'filename': local_filename,
            'size_mb': len(video_data) / 1024 / 1024
        }
        
        # Only save backups locally - don't upload intermediate videos to S3
        # Final video will be uploaded when recording session ends
        print(f"💾 Video backup saved locally only (S3 upload reserved for final video)")
        
        # Note: Removed automatic S3 upload for backups to avoid multiple videos in bucket
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error saving video backup: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Get the server IP address for easy access from mobile devices
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
        print(f"Server running at: https://{local_ip}:5000")
    except:
        print("Could not determine local IP address")
    
    # Run the app with SSL to enable camera access in browsers
    app.run(debug=False, host='0.0.0.0', ssl_context='adhoc') 