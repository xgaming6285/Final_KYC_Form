import os
import io
import cv2
import numpy as np
from google.cloud import vision

def test_vision_api():
    """Simple test to verify Google Cloud Vision API setup."""
    print("Testing Google Cloud Vision API connection...")
    
    # Check if credentials are set
    credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path:
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
        print("Please run setup_credentials.py first.")
        return False
    
    print(f"Using credentials from: {credentials_path}")
    
    try:
        # Create a client
        client = vision.ImageAnnotatorClient()
        
        # Create a simple image with text
        # We'll create a 500x200 white image with black text
        image = np.ones((200, 500, 3), dtype=np.uint8) * 255
        text = "Google Cloud Vision Test"
        cv2.putText(image, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save image temporarily
        cv2.imwrite("test_image.jpg", image)
        
        # Load the image into memory
        with io.open("test_image.jpg", 'rb') as image_file:
            content = image_file.read()
        
        # Create image object
        vision_image = vision.Image(content=content)
        
        # Perform text detection
        response = client.text_detection(image=vision_image)
        texts = response.text_annotations
        
        # Clean up test image
        os.remove("test_image.jpg")
        
        if texts:
            print("SUCCESS! API is working correctly.")
            print(f"Detected text: {texts[0].description.strip()}")
            return True
        else:
            print("API connected but no text was detected. This might be a configuration issue.")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    if test_vision_api():
        print("\nYour Google Cloud Vision API is set up correctly.")
        print("You can now run the main program with: python id_processor.py")
    else:
        print("\nThere was an issue with your Google Cloud Vision API setup.")
        print("Please check your credentials and API access permissions.") 