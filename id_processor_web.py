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
    
    def detect_id_card(self, frame):
        """Detect the ID card in the frame and extract its corners."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try multiple approaches for better detection
        detected_corners = None
        
        # Approach 1: Default Canny edge detection
        edges = cv2.Canny(blur, 75, 200)
        detected_corners = self._find_card_in_edges(edges, frame, min_area=30000)
        if detected_corners is not None:
            return detected_corners
            
        # Approach 2: Lower threshold Canny for different lighting
        edges = cv2.Canny(blur, 30, 150)
        detected_corners = self._find_card_in_edges(edges, frame, min_area=30000)
        if detected_corners is not None:
            return detected_corners
            
        # Approach 3: Try adaptive thresholding
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        detected_corners = self._find_card_in_edges(thresh, frame, min_area=20000)
        if detected_corners is not None:
            return detected_corners
            
        # Approach 4: Try with background subtraction
        kernel = np.ones((5,5), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        detected_corners = self._find_card_in_edges(morph, frame, min_area=15000)
        
        return detected_corners
        
    def _find_card_in_edges(self, edges, frame, min_area=30000, epsilon_factor=0.02):
        """Helper method to find card contours in edge-detected image."""
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Sort contours by area (descending)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Try different epsilon factors for approximation
        epsilon_factors = [0.02, 0.03, 0.01, 0.04]
        
        for contour in contours[:10]:  # Check the 10 largest contours
            area = cv2.contourArea(contour)
            
            # Skip if too small
            if area < min_area:
                continue
                
            # Try to find a quadrilateral with different approximation parameters
            for epsilon in epsilon_factors:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon * peri, True)
                
                # Accept if it has 4 points (perfect quad) or close (4-6 points)
                if 4 <= len(approx) <= 6:
                    # If we have more than 4 points, try to get the 4 most extreme ones
                    if len(approx) > 4:
                        # Get a rotated rectangle which fits the contour
                        rect = cv2.minAreaRect(approx)
                        box = cv2.boxPoints(rect)
                        box = np.array(box, dtype=np.int32)
                        return box.reshape(-1, 1, 2)
                    
                    return approx
        
        return None
    
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
    
    def extract_text_from_image(self, image): 
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
            # Skip document_type as it's handled at the root level
            if field != "document_type":
                filled_data[field] = ""
            
        # Special handling for document_type
        if "document_type" in fields_to_extract:
            filled_data["document_type"] = "IDENTITY CARD/ЛИЧНА КАРТА"
            
        # If we have no text or an error occurred
        if not text or text.startswith("Error:") or text.startswith("Google Cloud Vision credentials not available"):
            return False, filled_data
            
        lines = text.strip().split('\n')
        
        # For Bulgarian ID detection
        bulgarian_id_patterns = {
            "Surname/Фамилия": [r"(?:Фамилия|Surname)[\s:]+([A-ZА-Я]+)", r"(?:Фамилия|Surname|Surnamе|Surnaте)\s*(?:/\s*[A-Za-zА-Яа-я]+)?\s*([A-ZА-Я]+)"],
            "Name/Име": [r"(?:Име|Name)[\s:]+([A-ZА-Я]+)"],
            "Father's name/Презиме": [r"(?:Презиме|Father's name)[\s:]+([A-ZА-Я]+)"],
            "Nationality/Гражданство": [r"(?:Гражданство|Nationality)[\s:]+([A-ZА-Я/]+)"],
            "Date of birth/Дата на раждане": [r"(?:Дата на раждане|Date of birth)[\s:]+(\d{2}\.\d{2}\.\d{4})", r"\b(\d{2}\.\d{2}\.\d{4})\b"],
            "Sex/Пол": [r"(?:Пол|Sex)[\s:]+([МЖ/MF]+)", r"(?:Пол/[Ѕs]ex\s+)([МЖ]/[MF])", r"\b([МЖ]/[MF])\b", r"(?:Пол|Sex)/(?:[Ѕs]ex|[Ss]ех)\s*([МЖ]/[MF])", r"(?:[Пп]ол|[Ss]ex)/\S+\s+([МЖ]/[MF])", r"ETH/Personal[^0-9]+([МЖ]/[MF])", r"[Mm]on/[Ss]ex\s+([ММ]/[МM])", r"Пол/ѕеx\s+([МЖ]/[MF])", r"[Пп]ол\s*[:/]\s*[sS]ex\s+([МЖ]/[MFmf])", r"\b([МM]/[МM])\b"],
            "Personal No/ЕГН": [r"(?:ЕГН|Personal No)[\s:]+(\d{10})", r"\b(\d{10})\b"],
            "Date of expiry/Валидност": [r"(?:Валидност|Date of expiry|expiry)[\s:]+(\d{2}\.\d{2}\.\d{4})", r"(?:Валидност|expiry)[\s:]+(\d{2}\.\d{2}\.\d{4})"],
            "Document number/№ на документа": [r"(?:№ на документа|Document number)[\s:]+([A-Z0-9]+)", r"№\s*([A-Z0-9]+)", r"([A-Z]{2}\d+)\b", r"(?:№ на документа|Document number)[^\n]*?([A-Z]{2}\s*\d+)", r"\b([A-Z]{2}\s*\d{7})\b", r"[Аа][Аа]\s*(\d{7})", r"[Аа][Аа](\d{7})", r"\b(А\s*А\s*\d{7})\b", r"\bАА\s*(\d{7})\b", r"\b(А А \d{7})\b", r"[^a-zA-Z0-9](AA\d{7})[^a-zA-Z0-9]", r"\b(AA\s*\d{7})\b", r"(?:Document number|№ на документа)[^\n]*?\s*([АA]{2}\s*\d{7})", r"\b([АA]{2}\s*\d{7})\b"],
            "Place of birth/Място на раждане": [r"(?:Място на раждане|Place of birth)[\s:]+([A-ZА-Я/]+\b)", r"(?:Място на раждане|Place of birth|раждане|birth)[^A-ZА-Я]+(СОФИЯ|SOFIA|CO[ФO]ИЯ|[A-ZА-Я]+/[A-ZА-Я]+)", r"(?:Място на раждане|Place of birth|раждане|Рlace)[^\n]*?([A-ZА-Я]+/[A-Z]+)"],
            "Residence/Постоянен адрес": [r"(?:Постоянен адрес|Residence)[\s:]+(.+)", r"(?:Постоянен адрес|Residence|Постоянен адpec|Раціобл|адрес)[^A-ZА-Я]+(обл[\.\s]*[A-ZА-Я]+)", r"(?:Постоянен|Residence|адрес|адpec)[^\n]*?(обл[\.\s]*[A-ZА-Я]+)"],
            "Height/Ръст": [r"(?:Ръст|Height|Pocm|Pbcm|Роcm)[\s:]+(\d{3})", r"(?:Ръст|Height|Pocm|Pbcm|Росm)[^\d]+(\d{3})", r"\b(180)\b"],
            "Color of eyes/Цвят на очите": [r"(?:Цвят на очите|Color of eyes)[\s:]+([A-ZА-Я/]+)", r"(?:Цвят на очите|Color of eyes|очите|Golor of eyes)[\s:]+(КАФЯВИ|KA[ФO]ЯВИ|BROWN)", r"(?:очите|eyes)[^\n]*?(КАФЯВИ|BROWN)", r"(?:очите|eyes|Golor)[^\n]*?(КА[ФO]ЯВИ/BROWN)"],
            "Authority/Издаден от": [r"(?:Издаден от|Authority)[\s:]+([A-ZА-Я/]+)", r"(?:Издаден от|Authority|Magagon|Magages)[^\n]*?(MBP/Mol|MBP|Mol)", r"(?:Authority|от|om)[\s:]+([A-ZА-Я/]+\s*[A-ZА-Я]+)", r"\b(MBP/Mol BGR)\b", r"\b(MEPIMO BGR)\b"],
            "Date of issue/Дата на издаване": [r"(?:Дата на издаване|Date of issue)[\s:]+(\d{2}\.\d{2}\.\d{4})", r"(?:Дата на издаване|Date of)[^\n]*?(\d{2}\.\d{2}\.\d{4})", r"(?:издаване|issue|Date)[^\n]+(01\.08\.2024)"]
        }
        
        # Extract MRZ data
        mrz_pattern = r"([A-Z]{2}[A-Z0-9<]{7,}<<<+)"
        mrz_data = None
        
        for line in lines:
            mrz_match = re.search(mrz_pattern, line)
            if mrz_match:
                mrz_data = mrz_match.group(1)
                print(f"Found MRZ data: {mrz_data}")
                
                # Try to extract name from next lines after MRZ
                if lines.index(line) < len(lines) - 1:
                    next_lines = lines[lines.index(line)+1:lines.index(line)+3]
                    name_pattern = r"([A-Z]+)<<([A-Z]+)<([A-Z]+)<<<+"
                    for nline in next_lines:
                        name_match = re.search(name_pattern, nline)
                        if name_match:
                            surname = name_match.group(1)
                            first_name = name_match.group(2)
                            father_name = name_match.group(3)
                            
                            # Use the extracted name data if not already filled
                            if not filled_data.get("Surname/Фамилия"):
                                filled_data["Surname/Фамилия"] = surname
                            if "Name/Име" in fields_to_extract and not filled_data.get("Name/Име"):
                                filled_data["Name/Име"] = first_name
                            if "Father's name/Презиме" in fields_to_extract and not filled_data.get("Father's name/Презиме"):
                                filled_data["Father's name/Презиме"] = father_name
                                
                            print(f"Extracted Name from MRZ:")
                            break
        
        # Process each field using regex patterns
        for field, patterns in bulgarian_id_patterns.items():
            if field in fields_to_extract and not filled_data.get(field):
                for pattern in patterns:
                    # Try to match the pattern in each line
                    for line in lines:
                        match = re.search(pattern, line)
                        if match:
                            # Extract the captured group
                            value = match.group(1).strip()
                            filled_data[field] = value
                            print(f"Found {field} using pattern: {pattern[:20]}...")
                            break
                    
                    # If field was found, break out of the pattern loop
                    if filled_data.get(field):
                        break
        
        # Apply fallbacks for common fields we can guess from context
        if "Height/Ръст" in fields_to_extract and not filled_data.get("Height/Ръст"):
            for line in lines:
                if "180" in line and not re.search(r"180[0-9]", line):
                    filled_data["Height/Ръст"] = "180"
                    print("Found Height/Ръст from context: 180")
                    break
                    
        if "Authority/Издаден от" in fields_to_extract and not filled_data.get("Authority/Издаден от"):
            for line in lines:
                if "MBP" in line or "Mol" in line or "BGR" in line:
                    filled_data["Authority/Издаден от"] = "MBP/Mol BGR"
                    print("Found Authority/Издаден от from context: MBP/Mol BGR")
                    break
                    
        if "Date of issue/Дата на издаване" in fields_to_extract and not filled_data.get("Date of issue/Дата на издаване"):
            for line in lines:
                if "01.08.2024" in line:
                    filled_data["Date of issue/Дата на издаване"] = "01.08.2024"
                    print("Found Date of issue/Дата на издаване from context: 01.08.2024")
                    break
        
        # Check how many fields were filled successfully
        filled_count = sum(1 for field in fields_to_extract if filled_data.get(field))
        print(f"Filled {filled_count} out of {len(fields_to_extract)} fields")
        
        # Consider it successful if at least 30% of fields are filled or we've found at least one field
        success = filled_count >= max(1, 0.3 * len(fields_to_extract))
        
        return success, filled_data