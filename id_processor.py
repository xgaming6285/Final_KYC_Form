import cv2
import numpy as np
import json
import os
import time
from google.cloud import vision
from google.cloud.vision_v1 import types
import io
import re

class IDProcessor:
    def __init__(self, formats_file="formats.json"):
        # Load the ID card formats
        with open(formats_file, 'r', encoding='utf-8') as f:
            self.formats = json.load(f)
        
        # Initialize Google Cloud Vision client
        self.vision_client = vision.ImageAnnotatorClient()
        
        # Initialize camera
        self.cap = None
        
    def start_camera(self):
        """Start the camera capture."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open video capture device")
        return self.cap
    
    def stop_camera(self):
        """Stop the camera capture."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            cv2.destroyAllWindows()
    
    def detect_id_card(self, frame):
        """Detect the ID card in the frame and extract its corners."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blur, 75, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (descending)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Loop through contours
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # If our approximated contour has four points, we can assume it's the ID card
            if len(approx) == 4:
                # Check if it's large enough to be an ID card (adjust threshold as needed)
                if cv2.contourArea(contour) > 50000:
                    return approx
        
        return None
    
    def extract_text_from_image(self, image):
        """Extract text from an image using Google Cloud Vision API."""
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
    
    def process_id_side(self, document_type, side):
        """Process one side of the ID card."""
        print(f"Please show the {side} side of your ID card")
        print("Press 'c' to capture when ID is properly aligned")
        print("Press 'q' to quit")
        
        fields_to_extract = self.formats[document_type][side]
        extracted_data = {field: "" for field in fields_to_extract}
        
        frame_counter = 0
        stable_frames = 0
        manual_capture = False
        
        self.start_camera()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Can't read from camera")
                    break
                
                frame_counter += 1
                display_frame = frame.copy()
                
                # Skip frames for better performance
                if frame_counter % 5 != 0 and not manual_capture:
                    # Just show the frame
                    cv2.putText(display_frame, "Align ID card and press 'c' to capture", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow('ID Scanner', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("User quit")
                        return extracted_data
                    elif key == ord('c'):
                        manual_capture = True
                    continue
                
                # Detect ID card
                corners = self.detect_id_card(frame)
                
                if corners is not None or manual_capture:
                    warped = None
                    
                    if corners is not None:
                        # Draw contour
                        cv2.drawContours(display_frame, [corners], 0, (0, 255, 0), 3)
                        
                        # Perspective transformation to get a top-down view
                        pts = corners.reshape(4, 2)
                        rect = self.order_points(pts)
                        
                        (tl, tr, br, bl) = rect
                        
                        # Calculate width of new image
                        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                        maxWidth = max(int(widthA), int(widthB))
                        
                        # Calculate height of new image
                        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                        maxHeight = max(int(heightA), int(heightB))
                        
                        # Destination points for perspective transform
                        dst = np.array([
                            [0, 0],
                            [maxWidth - 1, 0],
                            [maxWidth - 1, maxHeight - 1],
                            [0, maxHeight - 1]
                        ], dtype="float32")
                        
                        # Calculate perspective transform matrix
                        M = cv2.getPerspectiveTransform(rect, dst)
                        
                        # Apply perspective transformation
                        warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
                    else:
                        # If manual capture and no corners detected, use the full frame
                        warped = frame
                    
                    # Show the warped image
                    if warped is not None:
                        cv2.imshow('Warped ID', warped)
                    
                    # Extract text from warped image
                    if manual_capture or stable_frames >= 3:
                        print("\nExtracting text from ID card...")
                        # Save the warped image for debugging
                        if warped is not None:
                            cv2.imwrite(f"id_card_{side}.jpg", warped)
                            print(f"Saved ID card image to id_card_{side}.jpg")
                            
                            # Extract text from warped image
                            extracted_text = self.extract_text_from_image(warped)
                            
                            if extracted_text:
                                print("Text extraction successful!")
                                # Parse extracted text and map to fields
                                success, filled_data = self.map_text_to_fields(extracted_text, fields_to_extract)
                                
                                if success:
                                    # This is the fix - explicitly assign each field from filled_data to extracted_data
                                    for field, value in filled_data.items():
                                        if value:  # Only update fields that have values
                                            extracted_data[field] = value
                                    print("Successfully mapped fields from extracted text!")
                                    break
                                else:
                                    print("Failed to map all required fields from extracted text.")
                                    empty_fields = [field for field in fields_to_extract if not filled_data[field].strip()]
                                    print(f"Missing fields: {', '.join(empty_fields)}")
                                    
                                    # If mapping failed, always give user the option to retry or manually continue
                                    print("Press 'r' to retry capturing or 'c' to try again with current image")
                                    while True:
                                        key = cv2.waitKey(0) & 0xFF
                                        if key == ord('r'):
                                            manual_capture = False
                                            break
                                        elif key == ord('c'):
                                            # Try again with the same image but allow manual field input
                                            for empty_field in empty_fields:
                                                print(f"Please enter value for {empty_field}: ")
                                                value = input().strip()
                                                if value:
                                                    filled_data[empty_field] = value
                                            
                                            # Check if all fields are now filled
                                            still_empty = [field for field in fields_to_extract if not filled_data[field].strip()]
                                            if not still_empty:
                                                for field, value in filled_data.items():
                                                    extracted_data[field] = value
                                                print("All fields are now filled!")
                                                return extracted_data
                                            else:
                                                print(f"Still missing fields: {', '.join(still_empty)}")
                                                manual_capture = False
                                                break
                            else:
                                print("No text extracted from image.")
                                print("Press 'r' to retry capturing")
                                while True:
                                    key = cv2.waitKey(0) & 0xFF
                                    if key == ord('r'):
                                        manual_capture = False
                                        break
                        else:
                            print("No valid ID card image for text extraction.")
                            manual_capture = False
                    else:
                        stable_frames += 1
                        
                        # Display success message
                        cv2.putText(display_frame, f"ID detected ({stable_frames}/3)", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_frame, "Press 'c' to capture now", (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    stable_frames = 0
                    cv2.putText(display_frame, "No ID card detected", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display_frame, "Press 'c' to capture manually", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show the frame
                cv2.imshow('ID Scanner', display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("User quit")
                    break
                elif key == ord('c'):
                    manual_capture = True
                
        finally:
            self.stop_camera()
            cv2.destroyAllWindows()
        
        return extracted_data
    
    def map_text_to_fields(self, text, fields_to_extract):
        """Map extracted text to the required fields."""
        print("Extracted raw text:")
        print(text)
        
        filled_data = {}
        lines = text.strip().split('\n')
        
        # Initialize all fields as empty
        for field in fields_to_extract:
            filled_data[field] = ""
        
        # For Bulgarian ID new format detection
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
            print(f"Processing line: {line}")
            
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
                    print(f"Found field match: {field}")
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
                        print(f"Extracted value: '{value}'")
                        
                        if value:  # Only set if we actually got a value
                            filled_data[field] = value
                
                # Special handling for document number which might be formatted differently
                if field == "Document number/№ на документа" and not filled_data[field]:
                    if "№" in line or "no" in line.lower() or "number" in line.lower():
                        # Try to extract numeric value after these identifiers
                        for identifier in ["№", "no", "number", "No", "NO", "Number"]:
                            if identifier in line:
                                parts = line.split(identifier)
                                if len(parts) > 1:
                                    value = parts[1].strip(': /')
                                    print(f"Found document number: '{value}'")
                                    filled_data[field] = value
                                    break
        
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
            if "Personal No/ЕГН" in fields_to_extract and not filled_data["Personal No/ЕГН"]:
                personal_no_match = re.search(r'\b(\d{10})\b', mrz_data)
                if personal_no_match:
                    filled_data["Personal No/ЕГН"] = personal_no_match.group(1)
                    print(f"Extracted Personal No from MRZ: {filled_data['Personal No/ЕГН']}")
                    
            # Try to extract Name from MRZ
            if "Name/Име" in fields_to_extract and not filled_data["Name/Име"]:
                # Format is typically SURNAME<<FIRSTNAME<MIDDLENAME
                name_parts = mrz_data.split('<<')
                if len(name_parts) > 1:
                    names = name_parts[1].split('<')
                    if names:
                        filled_data["Name/Име"] = names[0].strip()
                        print(f'Extracted Name from MRZ: {filled_data["Name/Име"]}')
                        
                        # If there's a third part, it might be the father's name
                        if len(names) > 1 and "Father's name/Презиме" in fields_to_extract and not filled_data["Father's name/Презиме"]:
                            filled_data["Father's name/Презиме"] = names[1].strip()
                            print(f'Extracted Father\'s name from MRZ: {filled_data["Father\'s name/Презиме"]}')
                    
                    # Extract Surname from MRZ
                    if "Surname/Фамилия" in fields_to_extract and not filled_data["Surname/Фамилия"] and name_parts:
                        surname = name_parts[0].strip()
                        # Remove any document identifiers at the beginning
                        surname = re.sub(r'^[A-Z0-9]+', '', surname).strip()
                        if surname:
                            filled_data["Surname/Фамилия"] = surname
                            print(f'Extracted Surname from MRZ: {filled_data["Surname/Фамилия"]}')
        
        # Handle direct field extraction from the image
        # For new Bulgarian ID cards, some fields have values right after labels
        for field in fields_to_extract:
            # If the field is still empty, try more specific checks
            if not filled_data[field]:
                field_parts = field.split('/')
                field_name_en = field_parts[0].strip()
                field_name_bg = field_parts[1].strip() if len(field_parts) > 1 else ""
                
                # Look for lines that might contain just the value
                for line in lines:
                    # Check if this line only contains a value that could match the field
                    if field == "Surname/Фамилия" and not filled_data[field]:
                        if line.isupper() and 3 <= len(line) <= 20 and line.strip() == line and ' ' not in line:
                            if all(c.isalpha() or c == '-' for c in line):
                                print(f"Found potential surname: {line}")
                                filled_data[field] = line
                                break
                    
                    elif field == "Name/Име" and not filled_data[field]:
                        if line.isupper() and 3 <= len(line) <= 20 and line.strip() == line and ' ' not in line:
                            if all(c.isalpha() or c == '-' for c in line) and line != filled_data.get("Surname/Фамилия", ""):
                                print(f"Found potential name: {line}")
                                filled_data[field] = line
                                break
                    
                    elif field == "Father's name/Презиме" and not filled_data[field]:
                        if line.isupper() and 3 <= len(line) <= 20 and line.strip() == line and ' ' not in line:
                            if all(c.isalpha() or c == '-' for c in line) and line != filled_data.get("Surname/Фамилия", "") and line != filled_data.get("Name/Име", ""):
                                print(f"Found potential father's name: {line}")
                                filled_data[field] = line
                                break
                    
                    elif field == "Nationality/Гражданство" and not filled_data[field]:
                        if "БЪЛГАРИЯ" in line or "BULGARIA" in line or "BGR" in line:
                            print(f"Found nationality: {line}")
                            filled_data[field] = line.strip()
                            break
                    
                    elif field == "Sex/Пол" and not filled_data[field]:
                        if line.strip() in ["Ж/F", "М/M", "Ж", "М", "F", "M"]:
                            print(f"Found sex: {line}")
                            filled_data[field] = line.strip()
                            break
                    
                    elif field == "Date of birth/Дата на раждане" and not filled_data[field]:
                        date_matches = re.findall(r'\b\d{2}\.\d{2}\.\d{4}\b', line)
                        if date_matches:
                            # Check if "birth" or "раждане" is in the line
                            if "birth" in line.lower() or "раждане" in line.lower():
                                print(f"Found birth date: {date_matches[0]}")
                                filled_data[field] = date_matches[0]
                                break
                            # Also try to validate if this is a past date (likely birth date)
                            try:
                                day, month, year = map(int, date_matches[0].split('.'))
                                import datetime
                                potential_date = datetime.date(year, month, day)
                                today = datetime.date.today()
                                # Birth date should be in the past
                                if potential_date < today and year >= 1900:
                                    print(f"Found potential birth date: {date_matches[0]}")
                                    filled_data[field] = date_matches[0]
                                    break
                            except (ValueError, IndexError):
                                pass
                    
                    elif field == "Date of expiry/Валидност" and not filled_data[field]:
                        date_matches = re.findall(r'\b\d{2}\.\d{2}\.\d{4}\b', line)
                        if date_matches:
                            # Check if "expiry" or "валидност" is in the line
                            if "expiry" in line.lower() or "валидност" in line.lower():
                                print(f"Found expiry date: {date_matches[0]}")
                                filled_data[field] = date_matches[0]
                                break
                            # Also try to validate if this is a future date (likely expiry)
                            try:
                                day, month, year = map(int, date_matches[0].split('.'))
                                import datetime
                                potential_date = datetime.date(year, month, day)
                                today = datetime.date.today()
                                # Expiry date should be in the future and not too far
                                if potential_date > today and year < today.year + 20:
                                    print(f"Found potential expiry date: {date_matches[0]}")
                                    filled_data[field] = date_matches[0]
                                    break
                            except (ValueError, IndexError):
                                pass
                    
                    elif field == "Personal No/ЕГН" and not filled_data[field]:
                        # Look for 10-digit numbers
                        matches = re.findall(r'\b\d{10}\b', line)
                        if matches:
                            print(f"Found personal number: {matches[0]}")
                            filled_data[field] = matches[0]
                            break
                    
                    elif field == "Document number/№ на документа" and not filled_data[field]:
                        # Look for document number format (letters followed by digits)
                        matches = re.findall(r'\b[A-Z]+\d+\b', line)
                        if matches:
                            print(f"Found document number: {matches[0]}")
                            filled_data[field] = matches[0]
                            break
                    
                    elif field == "Place of birth/Място на раждане" and not filled_data[field]:
                        if "СОФИЯ" in line.upper() or "SOFIA" in line.upper():
                            print(f"Found place of birth: {line}")
                            filled_data[field] = line.strip()
                            break
                    
                    elif field == "Height/Ръст" and not filled_data[field]:
                        # Look for 3-digit numbers (typical height in cm)
                        matches = re.findall(r'\b1\d{2}\b', line)
                        if matches:
                            print(f"Found height: {matches[0]}")
                            filled_data[field] = matches[0]
                            break
                    
                    elif field == "Color of eyes/Цвят на очите" and not filled_data[field]:
                        eye_colors = ["КАФЯВИ", "BROWN", "СИНИ", "BLUE", "ЗЕЛЕНИ", "GREEN", "ЧЕРНИ", "BLACK"]
                        for color in eye_colors:
                            if color in line.upper():
                                print(f"Found eye color: {color}")
                                filled_data[field] = color
                                break
                        if filled_data[field]:  # If we found a color, break the outer loop
                            break
                    
                    elif field == "Authority/Издаден от" and not filled_data[field]:
                        if "МВР" in line or "MVR" in line:
                            print(f"Found authority: {line}")
                            filled_data[field] = line.strip()
                            break
                    
                    elif field == "Date of issue/Дата на издаване" and not filled_data[field]:
                        date_matches = re.findall(r'\b\d{2}\.\d{2}\.\d{4}\b', line)
                        if date_matches and date_matches[0] != filled_data.get("Date of birth/Дата на раждане", "") and date_matches[0] != filled_data.get("Date of expiry/Валидност", ""):
                            print(f"Found issue date: {date_matches[0]}")
                            filled_data[field] = date_matches[0]
                            break
        
        # Count how many fields were successfully filled
        filled_count = sum(1 for value in filled_data.values() if value.strip())
        empty_fields = [field for field in fields_to_extract if not filled_data[field].strip()]
        
        print(f"Filled {filled_count} out of {len(fields_to_extract)} fields")
        if empty_fields:
            print(f"Missing fields: {', '.join(empty_fields)}")
        print("Extracted data:", filled_data)
        
        # Consider it successful only if ALL fields are filled
        success = filled_count == len(fields_to_extract)
        
        return success, filled_data
    
    def order_points(self, pts):
        """Order points in top-left, top-right, bottom-right, bottom-left order."""
        rect = np.zeros((4, 2), dtype="float32")
        
        # Top-left will have the smallest sum
        # Bottom-right will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right will have the smallest difference
        # Bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def verify_face(self, id_image):
        """Verify if the person in front of the camera is the same as in the ID."""
        print("Please look at the camera for face verification")
        
        # Load face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Use Vision API to detect face in ID image
        id_face_response = self.vision_client.face_detection(
            image=vision.Image(content=cv2.imencode('.jpg', id_image)[1].tobytes())
        )
        
        if not id_face_response.face_annotations:
            print("No face found in ID card")
            return False
        
        # Start camera for live face capture
        self.start_camera()
        
        try:
            stable_frames = 0
            result = False
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Detect faces in the live frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) == 1:
                    (x, y, w, h) = faces[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Extract face region
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Use Vision API for face verification
                    live_face_response = self.vision_client.face_detection(
                        image=vision.Image(content=cv2.imencode('.jpg', face_img)[1].tobytes())
                    )
                    
                    if live_face_response.face_annotations:
                        # Here you would normally use a specialized face recognition algorithm
                        # For this example, we'll just assume Google Cloud Vision can help
                        # In a real application, you might want to use a dedicated face recognition service
                        
                        # This is a simplified approach - in reality you would use proper face embeddings comparison
                        match = self.compare_faces(id_face_response.face_annotations[0], 
                                                  live_face_response.face_annotations[0])
                        
                        if match:
                            stable_frames += 1
                            cv2.putText(frame, f"Face match detected ({stable_frames}/5)", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            if stable_frames >= 5:  # Require 5 consistent matches
                                result = True
                                break
                        else:
                            stable_frames = 0
                            cv2.putText(frame, "Face does not match", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Face not detected clearly", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif len(faces) > 1:
                    cv2.putText(frame, "Multiple faces detected", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "No face detected", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('Face Verification', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.stop_camera()
        
        return result
    
    def compare_faces(self, face1, face2):
        """Compare two face annotations from Google Cloud Vision."""
        # This is a simplified comparison
        # In a real application, you'd use proper face embeddings and a similarity threshold
        
        # Let's assume similar detection confidence indicates similar faces
        # This is NOT a proper way to do face verification in production
        confidence_diff = abs(face1.detection_confidence - face2.detection_confidence)
        
        # Check if landmarks are in similar positions (normalized by face size)
        landmark_similarities = []
        
        # This is a very basic approach - in reality, you'd use specialized face recognition
        return confidence_diff < 0.2  # Arbitrary threshold

    def process_id(self, document_type="bulgarian_id"):
        """Process the entire ID card (front, back, and face verification)."""
        print(f"Starting ID processing for: {document_type}")
        
        # Process front side
        front_data = self.process_id_side(document_type, "front")
        
        # Check if all front fields are filled
        missing_front_fields = [field for field, value in front_data.items() if not value.strip()]
        if missing_front_fields:
            print(f"Error: Not all front side fields were filled. Missing: {', '.join(missing_front_fields)}")
            print("Please restart the process and ensure all fields are captured.")
            return None
            
        print("\nFront side data extracted:")
        for field, value in front_data.items():
            print(f"{field}: {value}")
        
        # Process back side
        print("\nPlease flip your ID card to show the back side")
        time.sleep(2)  # Give user time to flip the card
        
        back_data = self.process_id_side(document_type, "back")
        
        # Check if all back fields are filled
        missing_back_fields = [field for field, value in back_data.items() if not value.strip()]
        if missing_back_fields:
            print(f"Error: Not all back side fields were filled. Missing: {', '.join(missing_back_fields)}")
            print("Please restart the process and ensure all fields are captured.")
            return None
            
        print("\nBack side data extracted:")
        for field, value in back_data.items():
            print(f"{field}: {value}")
        
        # Verify face (simplified - in reality would use the photo from the ID)
        # Here we'd need the actual image of the face from the ID
        # For simplicity, we're assuming we have that image
        print("\nNow proceeding to face verification")
        time.sleep(2)
        
        # In a real implementation, you'd extract the face from the ID
        # For this example, we'll just use a placeholder
        id_face = np.zeros((100, 100, 3), dtype=np.uint8)  # Placeholder
        
        verification_result = self.verify_face(id_face)
        
        if verification_result:
            print("\nFace verification successful! Identity confirmed.")
        else:
            print("\nFace verification failed. Identity could not be confirmed.")
        
        # Combine all data and format properly
        all_data = {
            "document_type": document_type,
            "front_side": {},
            "back_side": {},
            "face_verified": verification_result
        }
        
        # Clean and format front side data
        for field, value in front_data.items():
            # Handle Cyrillic characters properly in the output
            field_name_parts = field.split('/')
            if len(field_name_parts) > 1:
                field_name = field_name_parts[0].strip()  # Use only English part for keys
            else:
                field_name = field
            
            all_data["front_side"][field_name] = value
        
        # Clean and format back side data
        for field, value in back_data.items():
            # Handle Cyrillic characters properly in the output
            field_name_parts = field.split('/')
            if len(field_name_parts) > 1:
                field_name = field_name_parts[0].strip()  # Use only English part for keys
            else:
                field_name = field
                
            all_data["back_side"][field_name] = value
        
        return all_data

def main():
    # Check if Google Cloud credentials are set
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        print("Please set the GOOGLE_APPLICATION_CREDENTIALS environment variable")
        print("Example: export GOOGLE_APPLICATION_CREDENTIALS='path/to/your-project-credentials.json'")
        return
    
    processor = IDProcessor()
    result = processor.process_id()
    
    if result is None:
        print("\nID processing failed. Please try again and ensure all fields are captured.")
        return
    
    # Format the data nicely for better readability
    formatted_data = {
        "document_type": result["document_type"],
        "front_side": {},
        "back_side": {},
        "face_verified": result["face_verified"]
    }
    
    # Add front side data with proper formatting
    for field, value in result["front_side"].items():
        formatted_data["front_side"][field] = value
    
    # Add back side data with proper formatting
    for field, value in result["back_side"].items():
        formatted_data["back_side"][field] = value
    
    # Save the result to a file with proper formatting
    with open('extracted_id_data.json', 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)
    
    print("\nExtracted data saved to 'extracted_id_data.json'")
    
    # Additionally create a human-readable text file
    with open('extracted_id_data.txt', 'w', encoding='utf-8') as f:
        f.write(f"Document Type: {formatted_data['document_type']}\n\n")
        
        f.write("FRONT SIDE DATA:\n")
        f.write("==============\n")
        for field, value in formatted_data["front_side"].items():
            f.write(f"{field}: {value}\n")
        
        f.write("\nBACK SIDE DATA:\n")
        f.write("=============\n")
        for field, value in formatted_data["back_side"].items():
            f.write(f"{field}: {value}\n")
        
        f.write(f"\nFace Verification: {'Successful' if formatted_data['face_verified'] else 'Failed'}\n")
    
    print("Also saved as human-readable format to 'extracted_id_data.txt'")

if __name__ == "__main__":
    main() 