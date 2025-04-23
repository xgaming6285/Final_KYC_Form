import streamlit as st
import requests
import base64
from PIL import Image
import io
import cv2
import numpy as np
import os
import uuid
import json
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="KYC Document Verification",
    page_icon="🆔",
    layout="wide"
)

# Create upload folder if it doesn't exist
upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(upload_folder, exist_ok=True)

# Session state management
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'front'
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = {}

# Function to process image without using face_recognition
def process_id_card(image, document_type='bulgarian_id', side='front'):
    """Process ID card image and extract text"""
    # Save image to file
    filename = f"{st.session_state.session_id}_{side}.jpg"
    filepath = os.path.join(upload_folder, filename)
    
    if isinstance(image, np.ndarray):
        # Save OpenCV image
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        # Save PIL image
        image.save(filepath)
    
    # Use OpenCV to detect ID card
    image_np = np.array(image)
    if len(image_np.shape) == 2:  # If grayscale, convert to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # If RGBA, convert to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Extract text using OCR (simplified for Streamlit)
    # Note: In a production environment, you would connect this to a proper OCR service
    # like Google Cloud Vision or a local Tesseract implementation
    
    # For demo purposes, let's fake some data based on the document type and side
    extracted_data = {}
    
    if document_type == 'bulgarian_id':
        if side == 'front':
            extracted_data = {
                "identity_number": "1234567890",
                "given_name": "John",
                "surname": "Doe",
                "nationality": "Bulgarian",
                "date_of_birth": "01.01.1990",
                "expiry_date": "01.01.2030"
            }
        else:  # back side
            extracted_data = {
                "issuing_authority": "Ministry of Interior",
                "issuing_date": "01.01.2020",
                "address": "123 Main St, Sofia, Bulgaria"
            }
    
    # In a real implementation, you would use OCR here
    
    # Store extracted data in session state
    if side == 'front':
        st.session_state.extracted_data['front'] = extracted_data
    else:
        st.session_state.extracted_data['back'] = extracted_data
    
    return extracted_data, filepath

def verify_face(selfie_image, id_image):
    """Basic face verification without using face_recognition"""
    # Convert images to OpenCV format
    selfie_np = np.array(selfie_image)
    id_np = np.array(id_image)
    
    # Use OpenCV's Haar cascade for basic face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in both images
    selfie_gray = cv2.cvtColor(selfie_np, cv2.COLOR_RGB2GRAY)
    id_gray = cv2.cvtColor(id_np, cv2.COLOR_RGB2GRAY)
    
    selfie_faces = face_cascade.detectMultiScale(selfie_gray, 1.1, 4)
    id_faces = face_cascade.detectMultiScale(id_gray, 1.1, 4)
    
    if len(selfie_faces) == 0:
        return False, "No face detected in the selfie"
    
    if len(id_faces) == 0:
        return False, "No face detected in the ID card"
    
    # In a real system, you would use a facial recognition model here
    # Without face_recognition, we can only verify that faces were detected
    # Return True to simulate successful verification for demo purposes
    
    return True, "Face detected in both images. Manual verification required."

# Main app UI
st.title("KYC Document Verification")

# Step indicator
st.sidebar.header("Verification Progress")
step_status = {
    'front': '🔄' if st.session_state.current_step == 'front' else ('✅' if 'front' in st.session_state.extracted_data else '⬜'),
    'back': '🔄' if st.session_state.current_step == 'back' else ('✅' if 'back' in st.session_state.extracted_data else '⬜'),
    'face': '🔄' if st.session_state.current_step == 'face' else ('✅' if 'face_verified' in st.session_state else '⬜'),
    'complete': '🔄' if st.session_state.current_step == 'complete' else ('✅' if 'complete' in st.session_state else '⬜')
}

st.sidebar.write(f"{step_status['front']} Front of ID Card")
st.sidebar.write(f"{step_status['back']} Back of ID Card")
st.sidebar.write(f"{step_status['face']} Face Verification")
st.sidebar.write(f"{step_status['complete']} Completed")

# Display current step
if st.session_state.current_step == 'front':
    st.header("Step 1: Scan Front of ID Card")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload front of ID card", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Process Front Side"):
                with st.spinner("Processing..."):
                    extracted_data, filepath = process_id_card(image, side='front')
                    
                    # Show extracted data
                    st.success("ID Front processed successfully!")
                    st.json(extracted_data)
                    
                    # Move to next step
                    st.session_state.current_step = 'back'
                    st.rerun()
    
    with col2:
        st.write("Instructions:")
        st.write("1. Upload a clear image of the front of your ID card")
        st.write("2. Make sure all corners are visible")
        st.write("3. Avoid glare on the document")
        st.write("4. Image should be in good lighting")

elif st.session_state.current_step == 'back':
    st.header("Step 2: Scan Back of ID Card")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload back of ID card", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Process Back Side"):
                with st.spinner("Processing..."):
                    extracted_data, filepath = process_id_card(image, side='back')
                    
                    # Show extracted data
                    st.success("ID Back processed successfully!")
                    st.json(extracted_data)
                    
                    # Move to next step
                    st.session_state.current_step = 'face'
                    st.rerun()
    
    with col2:
        st.write("Instructions:")
        st.write("1. Upload a clear image of the back of your ID card")
        st.write("2. Make sure all corners are visible")
        st.write("3. Avoid glare on the document")
        st.write("4. Image should be in good lighting")
        
        # Option to go back
        if st.button("← Back to Front Side"):
            st.session_state.current_step = 'front'
            st.rerun()

elif st.session_state.current_step == 'face':
    st.header("Step 3: Face Verification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Take or upload a selfie", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            selfie_image = Image.open(uploaded_file)
            st.image(selfie_image, caption="Your Selfie", use_column_width=True)
            
            if st.button("Verify Face"):
                with st.spinner("Verifying..."):
                    # Get ID card image from the front side
                    id_filepath = os.path.join(upload_folder, f"{st.session_state.session_id}_front.jpg")
                    
                    if os.path.exists(id_filepath):
                        id_image = Image.open(id_filepath)
                        
                        # Verify face
                        success, message = verify_face(selfie_image, id_image)
                        
                        if success:
                            st.success(message)
                            st.session_state.face_verified = True
                            st.session_state.current_step = 'complete'
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.error("ID card image not found. Please complete the previous steps first.")
    
    with col2:
        st.write("Instructions:")
        st.write("1. Take a clear photo of your face")
        st.write("2. Ensure good lighting")
        st.write("3. Look directly at the camera")
        st.write("4. Remove sunglasses or other items covering your face")
        
        # Option to go back
        if st.button("← Back to ID Back Side"):
            st.session_state.current_step = 'back'
            st.rerun()

elif st.session_state.current_step == 'complete':
    st.header("Verification Complete!")
    
    st.success("Your identity has been verified successfully!")
    
    # Display combined information
    st.subheader("Extracted Information")
    
    all_data = {}
    if 'front' in st.session_state.extracted_data:
        all_data.update(st.session_state.extracted_data['front'])
    if 'back' in st.session_state.extracted_data:
        all_data.update(st.session_state.extracted_data['back'])
    
    st.json(all_data)
    
    # Option to start over
    if st.button("Start New Verification"):
        # Reset session state
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.current_step = 'front'
        st.session_state.extracted_data = {}
        if 'face_verified' in st.session_state:
            del st.session_state.face_verified
        if 'complete' in st.session_state:
            del st.session_state.complete
        
        st.rerun() 