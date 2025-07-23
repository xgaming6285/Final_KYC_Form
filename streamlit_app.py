import streamlit as st
import os
import uuid
import cv2
import numpy as np
from PIL import Image
import io
import base64
from aws_face_recognition import face_locations, face_encodings, face_distance, compare_faces_aws
from s3_storage import upload_image_to_s3, get_s3_image_url, s3_storage
import sys
import json
import tempfile

# Import functionality from web_server.py
# Note: We're importing the module, not running the Flask app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to use our more robust IDProcessorWeb class, fall back to original if not available
try:
    from id_processor_web import IDProcessorWeb
    id_processor = IDProcessorWeb()
    st.sidebar.success("Using IDProcessorWeb with error handling")
except ImportError:
    try:
        from id_processor import IDProcessor
        id_processor = IDProcessor()
        st.sidebar.success("Using original IDProcessor")
    except ImportError:
        st.error("ID Processor module not found. Please check that id_processor.py is available.")
        st.stop()

# Create uploads directory for storing temporary files
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'streamlit_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_extracted_data(document_type, side, extracted_data):
    """Save the extracted data to a JSON file with timestamp."""
    import datetime
    # Create a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"extracted_data_{timestamp}.json"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # If file exists already, load its contents, otherwise initialize a new dictionary
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
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
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    return filename, filepath

def main():
    st.set_page_config(page_title="ID Card Processor", layout="wide")
    st.title("ID Card Processing System")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["ID Scanning", "Face Verification", "About"])
    
    # Initialize session state for tracking progress
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "front_processed" not in st.session_state:
        st.session_state.front_processed = False
    if "back_processed" not in st.session_state:
        st.session_state.back_processed = False
    if "front_data" not in st.session_state:
        st.session_state.front_data = {}
    if "back_data" not in st.session_state:
        st.session_state.back_data = {}
    if "front_image_s3_key" not in st.session_state:
        st.session_state.front_image_s3_key = None
    if "back_image_s3_key" not in st.session_state:
        st.session_state.back_image_s3_key = None
    
    # ID Scanning Page
    if page == "ID Scanning":
        st.header("ID Card Scanner")
        
        # Select document type
        document_type = st.selectbox("Select Document Type", 
                                    list(id_processor.formats.keys()),
                                    index=0)  # Default to first document type
        
        # Determine which side to scan
        if not st.session_state.front_processed:
            side = "front"
            st.info("Please scan the FRONT of your ID card")
        elif not st.session_state.back_processed:
            side = "back"
            st.info("Please scan the BACK of your ID card")
        else:
            side = st.radio("Select card side", ["front", "back"])
            if st.button("Reset Scanning Process"):
                st.session_state.front_processed = False
                st.session_state.back_processed = False
                st.session_state.front_data = {}
                st.session_state.back_data = {}
                st.session_state.front_image_s3_key = None
                st.session_state.back_image_s3_key = None
                st.experimental_rerun()
        
        # Upload image option
        uploaded_file = st.file_uploader("Upload ID card image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Upload image to S3 for processing
            s3_result = upload_image_to_s3(image, st.session_state.session_id, side)
            
            if not s3_result['success']:
                st.error(f"Failed to upload image to S3: {s3_result['error']}")
                return
            
            # Process the image
            fields_to_extract = id_processor.formats[document_type][side]
            extracted_text = id_processor.extract_text_from_image(image)
            
            if extracted_text:
                success, filled_data = id_processor.map_text_to_fields(extracted_text, fields_to_extract)
                
                if success:
                    st.success("ID card processed successfully!")
                    
                    # Save extracted data
                    filename, filepath = save_extracted_data(document_type, side, filled_data)
                    
                    # Display extracted information
                    st.subheader("Extracted Information")
                    for field, value in filled_data.items():
                        st.text_input(field, value)
                    
                    # Update session state
                    if side == "front":
                        st.session_state.front_processed = True
                        st.session_state.front_data = filled_data
                        st.session_state.front_image_s3_key = s3_result['s3_key']  # Store S3 key instead
                    else:
                        st.session_state.back_processed = True
                        st.session_state.back_data = filled_data
                        st.session_state.back_image_s3_key = s3_result['s3_key']  # Store S3 key instead
                    
                    # Next steps
                    if side == "front" and not st.session_state.back_processed:
                        st.info("Front side processed successfully. Please scan the back side of your ID card.")
                        if st.button("Continue to back side"):
                            st.experimental_rerun()
                    elif side == "back" or (st.session_state.front_processed and st.session_state.back_processed):
                        st.success("ID card fully processed! You can now proceed to face verification.")
                        if st.button("Continue to Face Verification"):
                            page = "Face Verification"
                            st.experimental_rerun()
                else:
                    st.warning("Partial text extraction. Some fields could not be filled.")
                    # Display partial results
                    st.subheader("Partial Results")
                    for field, value in filled_data.items():
                        st.text_input(field, value)
            else:
                st.error("Failed to extract text from image. Please try again with a clearer image.")
    
    # Face Verification Page
    elif page == "Face Verification":
        st.header("Face Verification")
        
        if not st.session_state.front_processed:
            st.warning("Please complete ID card scanning before proceeding to face verification.")
            if st.button("Go to ID Scanning"):
                page = "ID Scanning"
                st.experimental_rerun()
        else:
            st.subheader("Compare Live Face with ID Photo")
            
            # Get the ID image for face comparison (but don't display it)
            if st.session_state.front_image_s3_key:
                # Download the ID image from S3
                id_image_bytes = s3_storage.download_image(st.session_state.front_image_s3_key)
                if id_image_bytes:
                    # Convert bytes to image
                    id_image_array = np.frombuffer(id_image_bytes, np.uint8)
                    id_image = cv2.imdecode(id_image_array, cv2.IMREAD_COLOR)
                    id_image_rgb = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)
                else:
                    st.error("Failed to download ID image from S3")
                    return
                
                # Option for webcam capture
                st.subheader("Capture Your Face")
                picture = st.camera_input("Take a picture")
                
                if picture is not None:
                    # Process the captured image
                    bytes_data = picture.getvalue()
                    face_image = Image.open(io.BytesIO(bytes_data))
                    face_np = np.array(face_image)
                    
                    # Upload the captured face to S3
                    face_s3_result = upload_image_to_s3(face_image, st.session_state.session_id, 'face')
                    
                    if not face_s3_result['success']:
                        st.error(f"Failed to upload face image to S3: {face_s3_result['error']}")
                        return
                    
                    # Use AWS Rekognition to compare faces
                    try:
                        # Convert captured face to PIL Image for AWS Rekognition
                        face_image_pil = Image.fromarray(face_np)
                        
                        # Use AWS Rekognition to compare faces
                        comparison_result = compare_faces_aws(
                            source_image=id_image_rgb,  # ID card image
                            target_image=face_image_pil,  # Captured face
                            similarity_threshold=85  # Higher threshold for better security
                        )
                        
                        st.subheader("Verification Result")
                        
                        if comparison_result['success']:
                            similarity = comparison_result['similarity']
                            confidence = comparison_result['confidence']
                            is_match = comparison_result['match']
                            
                            if is_match:
                                st.success(f"✅ Face verification successful! Similarity: {similarity:.1f}%, Confidence: {confidence:.1f}%")
                                st.balloons()
                            else:
                                st.error(f"❌ Face verification failed. Similarity: {similarity:.1f}%, Confidence: {confidence:.1f}%")
                                
                        else:
                            error_msg = comparison_result.get('message', 'No matching faces found')
                            if 'error' in comparison_result:
                                error_msg = f"AWS Rekognition error: {comparison_result['error']}"
                            st.error(f"❌ Face verification failed. {error_msg}")
                            
                        # Removed image display - showing only the result as requested
                                
                    except Exception as e:
                        st.error(f"Error during AWS Rekognition face verification: {str(e)}")
                        
                        # Still display images for manual verification
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(id_image_rgb, caption="ID Card", width=300)
                        with col2:
                            st.image(face_np, caption="Captured Face", width=300)
            else:
                st.error("ID card image not found. Please complete the ID card scanning process first.")
    
    # About Page
    elif page == "About":
        st.header("About This Application")
        st.write("""
        This application provides a streamlined way to process and verify ID cards, with the following features:
        
        - ID card scanning and text extraction
        - Support for multiple document types
        - Face verification comparing live face with ID photo
        - Secure data handling
        
        This is the Streamlit version of the original web_server.py Flask application.
        """)
        
        st.subheader("Supported Document Types")
        for doc_type in id_processor.formats.keys():
            st.write(f"- {doc_type}")

if __name__ == "__main__":
    main() 