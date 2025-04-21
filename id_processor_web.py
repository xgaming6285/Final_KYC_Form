import cv2
import numpy as np
import json
import os
import time
from google.cloud import vision
from google.cloud.vision_v1 import types
import io
import re

class IDProcessorWeb:
    def __init__(self, formats_file="formats.json"):
        # Load the ID card formats
        with open(formats_file, 'r', encoding='utf-8') as f:
            self.formats = json.load(f)
        
        # Initialize Google Cloud Vision client with error handling
        try:
            self.vision_client = vision.ImageAnnotatorClient()
            self.vision_available = True
            print("Successfully initialized Google Cloud Vision client")
        except Exception as e:
            print(f"Warning: Could not initialize Google Cloud Vision client: {e}")
            print("OCR functionality will be limited")
            self.vision_client = None
            self.vision_available = False
        
        # Initialize camera
        self.cap = None
    
    def extract_text_from_image(self, image):
        """Extract text from an image using Google Cloud Vision API or fallback methods."""
        if not self.vision_available:
            print("Google Cloud Vision not available. Text extraction may not work.")
            return "Google Cloud Vision credentials not available. Cannot extract text."
        
        # Original implementation from IDProcessor
        try:
            # Apply image preprocessing to enhance text visibility
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding to improve text contrast
            thresholded = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,  # Block size
                2    # Constant subtracted from mean
            )
            
            # Apply morphological operations to remove noise
            kernel = np.ones((1, 1), np.uint8)
            morph = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
            
            # Try extracting text from both original and enhanced images
            results = []
            images_to_try = [image, thresholded, morph]
            
            for idx, img in enumerate(images_to_try):
                try:
                    # Convert numpy array to bytes
                    success, encoded_image = cv2.imencode('.jpg', img)
                    if not success:
                        continue
                    
                    content = encoded_image.tobytes()
                    
                    # Create image object
                    vision_image = vision.Image(content=content)
                    
                    # Perform text detection
                    response = self.vision_client.text_detection(image=vision_image)
                    texts = response.text_annotations
                    
                    if texts:
                        print(f"Successfully extracted text using image preprocessing method {idx}")
                        results.append(texts[0].description)
                except Exception as e:
                    print(f"Error extracting text from image {idx}: {e}")
            
            # Return the result with the most text
            if results:
                return max(results, key=len)
            
            # If all enhanced methods fail, try the original method
            try:
                # Convert numpy array to bytes
                success, encoded_image = cv2.imencode('.jpg', image)
                if not success:
                    return None
                
                content = encoded_image.tobytes()
                
                # Create image object
                vision_image = vision.Image(content=content)
                
                # Perform text detection with document text hint
                image_context = vision.ImageContext(language_hints=["bg", "en"])
                response = self.vision_client.document_text_detection(
                    image=vision_image,
                    image_context=image_context
                )
                
                if response.full_text_annotation:
                    return response.full_text_annotation.text
                
                if response.text_annotations:
                    return response.text_annotations[0].description
            except Exception as e:
                print(f"Error in document text detection: {e}")
            
            return None
        except Exception as e:
            print(f"Error in extract_text_from_image: {e}")
            return f"Error: {str(e)}"
    
    def map_text_to_fields(self, text, fields_to_extract):
        """Map extracted text to the required fields."""
        print("Extracted raw text:")
        print(text)
        
        filled_data = {}
        
        # Initialize all fields as empty
        for field in fields_to_extract:
            filled_data[field] = ""
            
        # If we have no text or an error occurred
        if not text or text.startswith("Error:") or text.startswith("Google Cloud Vision credentials not available"):
            return False, filled_data
            
        lines = text.strip().split('\n')
        
        # For Bulgarian ID detection
        bulgarian_id_patterns = {
            "Surname/Фамилия": [r"(?:Фамилия|Surname)[\s:]+([A-ZА-Я]+)"],
            "Name/Име": [r"(?:Име|Name)[\s:]+([A-ZА-Я]+)"],
            "Father's name/Презиме": [r"(?:Презиме|Father's name)[\s:]+([A-ZА-Я]+)"],
            "Nationality/Гражданство": [r"(?:Гражданство|Nationality)[\s:]+([A-ZА-Я/]+)"],
            "Date of birth/Дата на раждане": [r"(?:Дата на раждане|Date of birth)[\s:]+(\d{2}\.\d{2}\.\d{4})", r"\b(\d{2}\.\d{2}\.\d{4})\b"],
            "Sex/Пол": [r"(?:Пол|Sex)[\s:]+([МЖ/MF]+)"],
            "Personal No/ЕГН": [r"(?:ЕГН|Personal No)[\s:]+(\d{10})", r"\b(\d{10})\b"],
            "Date of expiry/Валидност": [r"(?:Валидност|Date of expiry|expiry)[\s:]+(\d{2}\.\d{2}\.\d{4})", r"(?:Валидност|expiry)[\s:]+(\d{2}\.\d{2}\.\d{4})"],
            "Document number/№ на документа": [r"(?:№ на документа|Document number)[\s:]+([A-Z0-9]+)", r"№\s*([A-Z0-9]+)"],
            "Place of birth/Място на раждане": [r"(?:Място на раждане|Place of birth)[\s:]+([A-ZА-Я]+)"],
            "Residence/Постоянен адрес": [r"(?:Постоянен адрес|Residence)[\s:]+(.+)"],
            "Height/Ръст": [r"(?:Ръст|Height)[\s:]+(\d{3})"],
            "Color of eyes/Цвят на очите": [r"(?:Цвят на очите|Color of eyes)[\s:]+([A-ZА-Я/]+)"],
            "Authority/Издаден от": [r"(?:Издаден от|Authority)[\s:]+([A-ZА-Я0-9/]+)"],
            "Date of issue/Дата на издаване": [r"(?:Дата на издаване|Date of issue)[\s:]+(\d{2}\.\d{2}\.\d{4})"]
        }
        
        # Check each line to see if it contains field information
        for line in lines:
            # Try pattern matching for Bulgarian ID fields
            for field, patterns in bulgarian_id_patterns.items():
                if field in fields_to_extract and not filled_data[field]:
                    for pattern in patterns:
                        matches = re.search(pattern, line, re.IGNORECASE)
                        if matches:
                            value = matches.group(1).strip()
                            print(f"Found {field} using pattern: {value}")
                            filled_data[field] = value
                            break
            
            # Traditional approach as fallback
            for field in fields_to_extract:
                if filled_data[field]:  # Skip if already filled by pattern matching
                    continue
                    
                # Extract field name without the translation part
                field_parts = field.split('/')
                field_name_en = field_parts[0].strip().lower()
                field_name_bg = field_parts[1].strip().lower() if len(field_parts) > 1 else ""
                
                # Convert line to lowercase for case-insensitive matching
                line_lower = line.lower()
                
                # Check if this line contains the field name in English or Bulgarian
                if field_name_en in line_lower or (field_name_bg and field_name_bg in line_lower):
                    # Try to extract the value after the field name
                    if field_name_en in line_lower:
                        parts = line_lower.split(field_name_en)
                        field_match = field_name_en
                    else:
                        parts = line_lower.split(field_name_bg)
                        field_match = field_name_bg
                    
                    if len(parts) > 1:
                        # Get the original case value
                        original_idx = line.lower().find(field_match) + len(field_match)
                        value = line[original_idx:].strip(': /')
                        
                        if value:  # Only set if we actually got a value
                            filled_data[field] = value
        
        # Look for Machine Readable Zone (MRZ) data at the bottom of the back side
        mrz_data = None
        for i, line in enumerate(lines):
            # Check for MRZ format (lines with <)
            if '<<' in line and line.isupper() and any(c.isdigit() for c in line):
                mrz_data = line
                # Check if there's more MRZ data in the next line
                if i+1 < len(lines) and '<<' in lines[i+1]:
                    mrz_data += ' ' + lines[i+1]
                break
                
        # Process MRZ data if found
        if mrz_data:
            print(f"Found MRZ data: {mrz_data}")
            
            # Try to extract Personal Number from MRZ
            if not filled_data.get("Personal No/ЕГН", ""):
                personal_no_match = re.search(r'\b(\d{10})\b', mrz_data)
                if personal_no_match:
                    filled_data["Personal No/ЕГН"] = personal_no_match.group(1)
                    print(f"Extracted Personal No from MRZ: {filled_data['Personal No/ЕГН']}")
                    
            # Try to extract Name from MRZ
            if not filled_data.get("Name/Име", ""):
                # Format is typically SURNAME<<FIRSTNAME<MIDDLENAME
                name_parts = mrz_data.split('<<')
                if len(name_parts) > 1:
                    names = name_parts[1].split('<')
                    if names:
                        filled_data["Name/Име"] = names[0].strip()
                        print(f'Extracted Name from MRZ: {filled_data["Name/Име"]}')
                        
                    # If there's a third part, it might be the father's name
                    if len(names) > 1 and not filled_data.get("Father's name/Презиме", ""):
                        filled_data["Father's name/Презиме"] = names[1].strip()
                        print(f'Extracted Father\'s name from MRZ: {filled_data["Father\'s name/Презиме"]}')
                
                # Extract Surname from MRZ
                if not filled_data.get("Surname/Фамилия", "") and name_parts:
                    surname = name_parts[0].strip()
                    # Remove any document identifiers at the beginning
                    surname = re.sub(r'^[A-Z0-9]+', '', surname).strip()
                    if surname:
                        filled_data["Surname/Фамилия"] = surname
                        print(f'Extracted Surname from MRZ: {filled_data["Surname/Фамилия"]}')
        
        # Count how many fields were successfully filled
        filled_count = sum(1 for value in filled_data.values() if value.strip())
        
        print(f"Filled {filled_count} out of {len(fields_to_extract)} fields")
        
        # Consider it successful if at least 30% of fields are filled or we've found at least one field
        success = filled_count >= max(1, 0.3 * len(fields_to_extract))
        
        return success, filled_data 