# Bulgarian ID Card Recognition System

This system is designed to extract information from Bulgarian ID cards, including the newer format introduced in 2024. It uses image processing and text recognition to extract personal data from both sides of Bulgarian ID cards.

## Features

- Extracts data from both sides of Bulgarian ID cards
- Supports new format Bulgarian ID cards 
- Captures the following information:
  - Document type
  - Full name (Surname, Name, Father's name)
  - Nationality
  - Date of birth
  - Sex/Gender
  - Personal Number (EGN)
  - Date of expiry
  - Document number
  - Place of birth
  - Residence address
  - Height
  - Eye color
  - Issuing authority
  - Date of issue
- Processes Machine Readable Zone (MRZ) data
- Utilizes Google Cloud Vision API for text recognition (when available)
- Can process from camera input or static images

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Google Cloud Vision API (for text extraction)

## Installation

1. Install required packages:
```
pip install -r requirements.txt
```

2. Set up Google Cloud Vision API credentials (for production use):
```
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your-project-credentials.json"
```

## Usage

### With Camera (Main Application)

```python
from id_processor import IDProcessor

# Initialize the processor
processor = IDProcessor()

# Process an ID card (will use camera)
result = processor.process_id(document_type="bulgarian_id")

# The result contains extracted data from both sides
print(result)
```

### With Static Images

```python
import cv2
from id_processor import IDProcessor

# Initialize the processor
processor = IDProcessor()

# Load the ID card images
front_image = cv2.imread('path_to_front_image.jpg')
back_image = cv2.imread('path_to_back_image.jpg')

# Extract text
front_text = processor.extract_text_from_image(front_image)
back_text = processor.extract_text_from_image(back_image)

# Map text to fields
front_fields = processor.formats['bulgarian_id']['front']
back_fields = processor.formats['bulgarian_id']['back']

success_front, front_data = processor.map_text_to_fields(front_text, front_fields)
success_back, back_data = processor.map_text_to_fields(back_text, back_fields)

# Combine the results
result = {
    "document_type": "bulgarian_id",
    "front": front_data,
    "back": back_data
}

print(result)
```

## Testing

The system includes a test script (`test_id.py`) that validates the text extraction functionality without requiring the Google Cloud Vision API. This is useful for development and testing purposes.

```
python test_id.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Web Interface for ID Scanning

This project now includes a web interface that allows you to scan and process ID cards directly from your phone or any web browser.

### Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure your Google Cloud Vision API credentials are set up correctly:
   ```
   python setup_credentials.py
   ```
   or on Windows:
   ```
   set_credentials.bat
   ```

3. Run the web server:
   ```
   python web_server.py
   ```

4. Access the web interface from:
   - Your computer: `http://localhost:5000`
   - Your phone (on the same network): `http://[your-computer-ip]:5000`
     - Find your computer's IP address using `ipconfig` on Windows or `ifconfig` on Linux/Mac
     - For example: `http://192.168.1.100:5000`

### Using the Web Interface

The web interface offers two main options:

1. **Upload ID Image**
   - Upload a previously captured image of an ID card
   - Select the document type and side (front/back)
   - Process the image and view the extracted information

2. **Use Camera**
   - Use your device's camera to capture an ID card in real-time
   - Works on mobile phones, tablets, and computers with cameras
   - Best results on phones with good cameras using the rear-facing camera

### Tips for Better Results

- Ensure good lighting when scanning ID cards
- Keep the ID card flat and avoid shadows
- Make sure the entire ID card is visible in the frame
- Hold the camera steady when capturing
- For best results, use the rear camera on mobile devices 

## Face Verification Feature

The system now includes face verification to compare the person's face with the photo on their ID card:

### How It Works

1. After scanning both sides of the ID card, you'll be automatically redirected to the face verification screen
2. Position your face within the indicated area and ensure good lighting
3. Click the "Capture" button to take a photo of your face
4. Click "Verify Face" to compare your face with the ID photo
5. The system will analyze both images and determine if they match
6. Results will show a comparison of both images and verification status

### Requirements for Face Verification

- `face_recognition` package (automatically installed with requirements.txt)
- Proper lighting for accurate face comparison
- Clear visibility of your face when capturing

### Tips for Successful Face Verification

- Ensure good, even lighting on your face
- Remove glasses, hats, or other accessories that might interfere with recognition
- Try to match the angle and expression of the photo on your ID
- For best results, use a front-facing camera in good lighting conditions

### Privacy Notice

- Face images are processed locally on the server and not stored permanently
- Only the verification result is saved to complete the KYC process 