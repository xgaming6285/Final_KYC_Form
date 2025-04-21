from flask import Flask, request, jsonify, render_template, redirect, url_for
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
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_id', methods=['POST'])
def process_id():
    if 'file' not in request.files and 'image_data' not in request.form:
        return jsonify({'error': 'No file or image data provided'}), 400
    
    result = {}
    document_type = request.form.get('document_type', 'bulgarian_id')
    side = request.form.get('side', 'front')
    
    # Process uploaded file
    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Save file with unique name
            filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
                    'extracted_text': extracted_text
                }
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
                    'extracted_text': extracted_text
                }
            else:
                result = {
                    'success': False,
                    'error': 'Failed to extract text from image'
                }
        except Exception as e:
            return jsonify({'error': f'Error processing image data: {str(e)}'}), 400
    
    return jsonify(result)

@app.route('/detect', methods=['GET'])
def detect_page():
    return render_template('detect.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

if __name__ == '__main__':
    # Run the app with SSL to enable camera access in browsers
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc') 