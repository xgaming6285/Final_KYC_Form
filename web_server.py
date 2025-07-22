from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
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
from aws_face_recognition import face_locations, face_encodings, face_distance, compare_faces_aws
from image_quality_validator import ImageQualityValidator

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

app = Flask(__name__)
# Use absolute path for uploads directory
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
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
    filename = f"{session_id}_{side}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Process uploaded file
    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Save file with consistent naming
            file.save(filepath)
            
            # Process the image
            image = cv2.imread(filepath)
            
            if image is None:
                return jsonify({'error': 'Could not read image file'}), 400
            
            # Validate image quality instead of extracting text
            validation_result = quality_validator.validate_id_image(image)
            feedback = quality_validator.get_quality_feedback(validation_result)
            
            if validation_result['is_valid']:
                result = {
                    'success': True,
                    'validation': validation_result,
                    'feedback': feedback,
                    'image_path': filepath,
                    'session_id': session_id,
                    'message': f'ID {side} side captured successfully!',
                    'quality_score': int(validation_result['quality_score'])
                }
            
            # Add next_step field to redirect to back side when front is complete
            if side == 'front':
                result['next_step'] = 'back'
            elif side == 'back':
                result['next_step'] = 'face'
        else:
            result = {
                'success': False,
                'validation': validation_result,
                'feedback': feedback,
                'error': feedback['message'],
                'recommendations': feedback.get('recommendations', ''),
                'quality_score': int(validation_result['quality_score'])
            }
    
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
            
            # Save the captured image
            image_pil.save(filepath)
            print(f"Saved captured image to {filepath}")
            
            # Convert PIL image to OpenCV format
            image_np = np.array(image_pil)
            image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Validate image quality instead of extracting text
            validation_result = quality_validator.validate_id_image(image)
            feedback = quality_validator.get_quality_feedback(validation_result)
            
            if validation_result['is_valid']:
                result = {
                    'success': True,
                    'validation': validation_result,
                    'feedback': feedback,
                    'image_path': filepath,
                    'session_id': session_id,
                    'message': f'ID {side} side captured successfully!',
                    'quality_score': int(validation_result['quality_score'])
                }
                
                # Add next_step field to redirect to back side when front is complete
                if side == 'front':
                    result['next_step'] = 'back'
                elif side == 'back':
                    result['next_step'] = 'face'
            else:
                result = {
                    'success': False,
                    'validation': validation_result,
                    'feedback': feedback,
                    'error': feedback['message'],
                    'recommendations': feedback.get('recommendations', ''),
                    'quality_score': int(validation_result['quality_score'])
                }
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
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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
        
        # Save the captured face image
        face_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_face.jpg")
        face_image.save(face_filepath)
        
        # Convert to a format usable by face_recognition
        face_image_np = np.array(face_image)
        face_image_rgb = cv2.cvtColor(face_image_np, cv2.COLOR_BGR2RGB)
        
        # Find the ID card front image path - prioritize the complete processed image
        id_front_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_front_complete.jpg")
        
        # If complete image doesn't exist, check for the original front image
        if not os.path.exists(id_front_path):
            id_front_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_front.jpg")
            
        if not os.path.exists(id_front_path):
            return jsonify({
                'success': False,
                'message': 'ID card image not found. Please complete the ID card scanning process first.'
            }), 400
        
        # Load the ID card image
        id_image = cv2.imread(id_front_path)
        id_image_rgb = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)
        
        # For debugging and display purposes
        print(f"Using ID image from: {id_front_path}")
        
        # *** IMPORTANT: Use the full ID card image for comparison rather than extracting face ***
        # This ensures we always have the ID photo visible for comparison
        id_photo_url = f"/uploads/{os.path.basename(id_front_path)}"
        
        # Use AWS Rekognition to compare faces directly
        try:
            # Convert face image to format compatible with AWS Rekognition  
            face_image_pil = Image.fromarray(cv2.cvtColor(face_image_np, cv2.COLOR_BGR2RGB))
            
            # Use AWS Rekognition to compare the captured face with the ID card image
            comparison_result = compare_faces_aws(
                source_image=id_image_rgb,  # ID card image
                target_image=face_image_pil,  # Captured face
                similarity_threshold=85  # Higher threshold for better security
            )
            
            print(f"AWS Rekognition comparison result: {comparison_result}")
            
            if comparison_result['success']:
                similarity = comparison_result['similarity']
                confidence = comparison_result['confidence']
                is_match = comparison_result['match']
                face_distance = comparison_result['face_distance']
                
                # Try to extract and save face region from ID for display purposes
                try:
                    from aws_face_recognition import aws_face_recognition
                    id_face_region = aws_face_recognition.extract_face_region(id_image_rgb)
                    if id_face_region is not None:
                        id_face_pil = Image.fromarray(cv2.cvtColor(id_face_region, cv2.COLOR_BGR2RGB))
                        id_face_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_id_face_detected.jpg")
                        id_face_pil.save(id_face_path)
                        id_display_url = f"/uploads/{session_id}_id_face_detected.jpg"
                    else:
                        id_display_url = id_photo_url
                except Exception as e:
                    print(f"Error extracting face for display: {str(e)}")
                    id_display_url = id_photo_url
                
                return jsonify({
                    'success': is_match,
                    'message': f'Face verification {"successful" if is_match else "failed"}. Similarity: {similarity:.1f}%, Confidence: {confidence:.1f}%' if is_match else 
                               f'Face verification failed. The captured face does not match the ID card photo. Similarity: {similarity:.1f}%',
                    'face_distance': float(face_distance),
                    'similarity': float(similarity),
                    'confidence': float(confidence),
                    'id_photo_url': id_display_url
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
                    'confidence': 0.0,
                    'id_photo_url': id_photo_url
                })
                
        except Exception as e:
            print(f"Error in AWS Rekognition face comparison: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Face verification could not be completed. AWS Rekognition error: {str(e)}',
                'face_distance': 1.0,
                'id_photo_url': id_photo_url
            })
    
    except Exception as e:
        print(f"Error in face verification: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error during face verification: {str(e)}'
        }), 500

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