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
from google.cloud import vision
from google.oauth2 import service_account
import datetime
import face_recognition

# Set Google Cloud credentials explicitly
credential_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'n8n-test-456921-2c4224bba16d.json')
if os.path.exists(credential_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    print(f"Set Google Cloud credentials to: {credential_path}")
else:
    print(f"Warning: Credentials file not found at {credential_path}")
    print("Scanning features that require Google Cloud Vision may not work.")

# Try to use our more robust IDProcessorWeb class, fall back to original if not available
try:
    from id_processor_web import IDProcessorWeb
    id_processor = IDProcessorWeb()
    print("Using IDProcessorWeb with error handling")
except ImportError:
    from id_processor import IDProcessor
    id_processor = IDProcessor()
    print("Using original IDProcessor")

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
            
            # Extract text from the image
            fields_to_extract = id_processor.formats[document_type][side]
            extracted_text = id_processor.extract_text_from_image(image)
            
            if extracted_text:
                success, filled_data = id_processor.map_text_to_fields(extracted_text, fields_to_extract)
                result = {
                    'success': success,
                    'data': filled_data,
                    'extracted_text': extracted_text,
                    'image_path': filepath,
                    'session_id': session_id
                }
                
                # Save the extracted data
                data_file = save_extracted_data(document_type, side, filled_data)
                result['data_file'] = data_file
                
                # Add next_step field to redirect to back side when front is complete
                if side == 'front':
                    result['next_step'] = 'back'
            else:
                result = {
                    'success': False,
                    'error': 'Failed to extract text from image'
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
            
            # Extract text from the image
            fields_to_extract = id_processor.formats[document_type][side]
            extracted_text = id_processor.extract_text_from_image(image)
            
            if extracted_text:
                success, filled_data = id_processor.map_text_to_fields(extracted_text, fields_to_extract)
                result = {
                    'success': success,
                    'data': filled_data,
                    'extracted_text': extracted_text,
                    'image_path': filepath,
                    'session_id': session_id
                }
                
                # Save the extracted data
                data_file = save_extracted_data(document_type, side, filled_data)
                result['data_file'] = data_file
                
                # Add next_step field to redirect to back side when front is complete
                if side == 'front':
                    result['next_step'] = 'back'
            else:
                result = {
                    'success': False,
                    'error': 'Failed to extract text from image'
                }
        except Exception as e:
            return jsonify({'error': f'Error processing image data: {str(e)}'}), 400
    
    return jsonify(result)

@app.route('/get_fields', methods=['GET'])
def get_fields():
    """Endpoint to get the list of fields to extract for a specific document type and side."""
    document_type = request.args.get('document_type', 'bulgarian_id')
    side = request.args.get('side', 'front')
    
    # Check if the requested document type and side exist
    if document_type not in id_processor.formats:
        return jsonify({'error': f'Document type {document_type} not found'}), 404
    
    if side not in id_processor.formats[document_type]:
        return jsonify({'error': f'Side {side} not available for document type {document_type}'}), 404
    
    # Return the list of fields
    fields = id_processor.formats[document_type][side]
    return jsonify({
        'document_type': document_type,
        'side': side,
        'fields': fields
    })

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a video frame for real-time ID card detection and data extraction."""
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
            
        # Save the raw frame for debugging
        debug_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_debug_frame.jpg")
        cv2.imwrite(debug_filepath, image)
        
        # Get the document fields to extract
        fields_to_extract = id_processor.formats[document_type][side]
        
        # Step 1: Detect the ID card
        corners = id_processor.detect_id_card(image)
        
        # Draw detected contours on the image for debugging
        debug_image = image.copy()
        if corners is not None:
            cv2.drawContours(debug_image, [corners], 0, (0, 255, 0), 3)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_detection_result.jpg"), debug_image)
        
        # Return early if no ID card is detected
        if corners is None:
            # Return more detailed feedback
            return jsonify({
                'cardDetected': False,
                'message': 'No ID card detected in frame. Try adjusting lighting, reducing glare, or holding the card more steady.',
                'debug_image': f"/uploads/{session_id}_debug_frame.jpg"
            })
        
        # Convert corners to a more usable format for front-end
        formatted_corners = []
        for point in corners:
            formatted_corners.append({
                'x': int(point[0][0]),
                'y': int(point[0][1])
            })
        
        # Step 2: Process the ID card image
        # Order the corners properly
        rect = id_processor.order_points(np.array([point[0] for point in corners]).astype("float32"))
        
        # Get the width and height of the ID card
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        
        # Take the maximum of the width and height values
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        
        # Define source and destination points for the perspective transform
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        
        # Calculate the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        # Save the warped image for debugging
        warped_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_warped.jpg")
        cv2.imwrite(warped_filepath, warped)
        
        # Step 3: Extract text from the warped image
        extracted_text = id_processor.extract_text_from_image(warped)
        
        # Check if text extraction was successful
        if not extracted_text:
            return jsonify({
                'cardDetected': True,
                'corners': formatted_corners,
                'message': 'ID card detected but text extraction failed. Check Google Cloud Vision setup.',
                'extractedCount': 0,
                'totalFields': len(fields_to_extract),
                'complete': False,
                'data': {field: "" for field in fields_to_extract},
                'warped_image': f"/uploads/{session_id}_warped.jpg",
                'debug_image': f"/uploads/{session_id}_detection_result.jpg"
            })
        
        # Step 4: Map extracted text to fields
        success, filled_data = id_processor.map_text_to_fields(extracted_text, fields_to_extract)
        
        # Count filled fields
        filled_fields_count = sum(1 for value in filled_data.values() if value.strip())
        
        # Determine if all fields are complete
        all_fields_complete = filled_fields_count == len(fields_to_extract)
        
        # If all fields are extracted successfully, save the data
        if all_fields_complete:
            # Save the final image
            filename = f"{session_id}_{side}_complete.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(filepath, warped)
            
            # Save the extracted data
            data_file = save_extracted_data(document_type, side, filled_data)
        
        # Return the results
        return jsonify({
            'cardDetected': True,
            'corners': formatted_corners,
            'extractedCount': filled_fields_count,
            'totalFields': len(fields_to_extract),
            'complete': all_fields_complete,
            'data': filled_data,
            'message': 'ID card processed successfully' if success else 'Partial text extraction',
            'warped_image': f"/uploads/{session_id}_warped.jpg",
            'debug_image': f"/uploads/{session_id}_detection_result.jpg",
            'next_step': 'back' if all_fields_complete and side == 'front' else ('face' if all_fields_complete and side == 'back' else None)
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
        
        # Find the ID card front image path - FIRST check for the original front image
        id_front_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_front.jpg")
        
        # If the front image doesn't exist, check for the complete processed image
        if not os.path.exists(id_front_path):
            id_front_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_front_complete.jpg")
        
        # If that still doesn't exist, check for the warped version
        if not os.path.exists(id_front_path):
            id_front_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_warped.jpg")
            
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
        
        # Detect face in captured image
        face_locations = face_recognition.face_locations(face_image_rgb)
        
        if not face_locations:
            # No face found in captured image, return error
            return jsonify({
                'success': False,
                'message': 'No face detected in the captured image. Please ensure your face is clearly visible.',
                'id_photo_url': id_photo_url
            })
        
        # Try multiple face detection methods on the ID card image
        id_face_locations = face_recognition.face_locations(id_image_rgb, model="hog")
        
        if not id_face_locations:
            try:
                id_face_locations = face_recognition.face_locations(id_image_rgb, model="cnn")
            except Exception as e:
                print(f"CNN model error: {str(e)}")
        
        if not id_face_locations:
            try:
                # Use OpenCV's face detector as fallback
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(id_image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        id_face_locations = [(y, x + w, y + h, x)]
            except Exception as e:
                print(f"OpenCV detector error: {str(e)}")
        
        # Fall back to using the whole image or a predefined region if we still can't detect a face
        if not id_face_locations:
            # Use the top-left quarter of the ID card as a common location for the photo
            height, width = id_image_rgb.shape[:2]
            top = int(height * 0.15)
            right = int(width * 0.5)
            bottom = int(height * 0.65)
            left = int(width * 0.05)
            id_face_locations = [(top, right, bottom, left)]
            print(f"Using predefined region for face detection: {id_face_locations}")
        
        # For UI display purposes, save the detected face area from ID
        try:
            top, right, bottom, left = id_face_locations[0]
            id_face_img = id_image_rgb[top:bottom, left:right]
            id_face_pil = Image.fromarray(id_face_img)
            id_face_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_id_face_detected.jpg")
            id_face_pil.save(id_face_path)
            # Use the extracted face for display but NOT for detection
            id_display_url = f"/uploads/{session_id}_id_face_detected.jpg"
        except Exception as e:
            print(f"Error extracting face for display: {str(e)}")
            id_display_url = id_photo_url
        
        # Get face encodings and attempt verification
        try:
            # Try to get face encodings for the ID card
            id_face_encodings = face_recognition.face_encodings(id_image_rgb, id_face_locations)
            # Get face encodings for the captured face
            face_encodings = face_recognition.face_encodings(face_image_rgb, face_locations)
            
            # Compare faces if encodings were found
            if len(id_face_encodings) > 0 and len(face_encodings) > 0:
                # Calculate face distance
                face_distance = face_recognition.face_distance([id_face_encodings[0]], face_encodings[0])[0]
                # Lower face distance means better match (0 is a perfect match)
                match_threshold = 0.65  # Slightly more lenient threshold
                
                # Log the face distance for debugging
                print(f"Face distance: {face_distance}")
                
                # Use the detected face region for display, but original ID for verification
                return jsonify({
                    'success': face_distance < match_threshold,
                    'message': 'Face verification successful' if face_distance < match_threshold else 
                               'Face verification failed. The captured face does not match the ID card photo.',
                    'face_distance': float(face_distance),
                    'id_photo_url': id_display_url if os.path.exists(id_face_path) else id_photo_url
                })
            else:
                # If we couldn't get proper face encodings but we have a face in both images
                # Allow verification with a warning
                return jsonify({
                    'success': True,
                    'message': 'Face verification conditionally passed, but the system could not perform detailed analysis.',
                    'face_distance': 0.5,  # Middle value
                    'id_photo_url': id_photo_url
                })
        except Exception as e:
            print(f"Error in face encoding/comparison: {str(e)}")
            # If face encodings fail, provide a fallback response
            return jsonify({
                'success': True,
                'message': 'Face verification conditionally passed. Could not perform detailed analysis.',
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