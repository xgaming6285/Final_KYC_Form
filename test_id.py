import json
import re

# Sample text from a Bulgarian ID card (front side)
sample_front_text = """
РЕПУБЛИКА БЪЛГАРИЯ
ЛИЧНА КАРТА
REPUBLIC OF BULGARIA
IDENTITY CARD

Фамилия ИВАНОВА
Surname IVANOVA
Име СЛАВИНА
Name SLAVINA
Презиме ГЕОРГИЕВА
Father's name GEORGIEVA
Гражданство БЪЛГАРИЯ/BGR
Nationality
Дата на раждане 01.08.1995
Date of birth
Пол Ж/F ЕГН/Personal No 9508010133
Sex
Валидност 17.06.2034
Date of expiry
№ на документа AA0000000
Document number
"""

# Sample text from a Bulgarian ID card (back side)
sample_back_text = """
Фамилия/Surname ИВАНОВА
Място на раждане/Place of birth СОФИЯ/SOFIA
Постоянен адрес/Residence ОБЛ.СОФИЯ
общ.СТОЛИЧНА гр.СОФИЯ/SOFIA
бул.КНЯГИНЯ МАРИЯ ЛУИЗА 48 ет.5 ап.26
Ръст/Height 168 Цвят на очите/Color of eyes КАФЯВИ/BROWN
Издаден от/Authority МВР/MoI BGR
Дата на издаване/Date of issue 17.06.2024

IDBGRAA0000000<<<<<<<<<<<<<<<
9508015F3406175BGR9508010133<0
IVANOVA<<SLAVINA<GEORGIEVA<<<<
"""

# Load the ID formats
with open('formats.json', 'r', encoding='utf-8') as f:
    formats = json.load(f)

# Define a simplified map_text_to_fields function based on the updated implementation
def map_text_to_fields(text, fields_to_extract):
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
        "Sex/Пол": [r"(?:Пол|Sex)[\s:]+([МЖ/MF]+)", r"(?:Пол/[Ss]ex\s+)([МЖ]/[MF])", r"\b([МЖ]/[MF])\b", r"(?:Пол|Sex)/(?:[Ѕs]ex|[Ss]ех)\s*([МЖ]/[MF])", r"(?:[Пп]ол|[Ss]ex)/\S+\s+([МЖ]/[MF])", r"ETH/Personal[^0-9]+([МЖ]/[MF])", r"[Mm]on/[Ss]ex\s+([МM]/[МM])", r"Пол/ѕеx\s+([МЖ]/[МM])", r"[Пп]ол\s*[:/]\s*[sS]ex\s+([МЖ]/[MFmf])", r"\b([МM]/[МM])\b"],
        "Personal No/ЕГН": [r"(?:ЕГН|Personal No)[\s:]+(\d{10})", r"\b(\d{10})\b"],
        "Date of expiry/Валидност": [r"(?:Валидност|Date of expiry|expiry)[\s:]+(\d{2}\.\d{2}\.\d{4})", r"(?:Валидност|expiry)[\s:]+(\d{2}\.\d{2}\.\d{4})"],
        "Document number/№ на документа": [r"(?:№ на документа|Document number)[\s:]+([A-Z0-9]+)", r"№\s*([A-Z0-9]+)", r"([A-Z]{2}\d+)\b", r"(?:№ на документа|Document number)[^\n]*?([A-Z]{2}\s*\d+)", r"\b([A-Z]{2}\s*\d{7})\b", r"[Аа][Аа]\s*(\d{7})", r"[Аа][Аа](\d{7})", r"\b(А\s*А\s*\d{7})\b", r"\bАА\s*(\d{7})\b", r"\b(А А \d{7})\b", r"[^a-zA-Z0-9](AA\d{7})[^a-zA-Z0-9]", r"\b(AA\s*\d{7})\b", r"(?:Document number|№ на документа)[^\n]*?\s*([АA]{2}\s*\d{7})", r"\b([АA]{2}\s*\d{7})\b"],
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
    
    # Add special handling for document number
    if "Document number/№ на документа" in fields_to_extract and not filled_data["Document number/№ на документа"]:
        # Look for patterns like AA1234567 (2 letters followed by digits)
        # This handles both Latin and Cyrillic characters
        doc_number_patterns = [
            r'\b([A-Z]{2}\d{7})\b',
            r'\b([АA][АA]\d{7})\b',  # Mixed Cyrillic/Latin
            r'\b(А\s*А\s*\d{7})\b',  # Cyrillic with spaces
            r'\b(A\s*A\s*\d{7})\b',  # Latin with spaces
            r'\b(АА\d{7})\b',        # Cyrillic
            r'\b(AA\d{7})\b'         # Latin
        ]
        
        for pattern in doc_number_patterns:
            for line in lines:
                matches = re.search(pattern, line)
                if matches:
                    doc_number = matches.group(1)
                    # Standardize to Latin AA format
                    doc_number = doc_number.replace('А', 'A').replace(' ', '')
                    print(f"Found Document number using standalone pattern: {doc_number}")
                    filled_data["Document number/№ на документа"] = doc_number
                    break
            if filled_data["Document number/№ на документа"]:
                break
                
    # Count how many fields were successfully filled
    filled_count = sum(1 for value in filled_data.values() if value.strip())
    
    print(f"Filled {filled_count} out of {len(fields_to_extract)} fields")
    print("Extracted data:", filled_data)
    
    # Consider it successful if at least 30% of fields are filled
    success = filled_count >= 0.3 * len(fields_to_extract)
    
    return success, filled_data

# Process front side
print("TESTING FRONT SIDE EXTRACTION:")
front_fields = formats['bulgarian_id']['front']
success, front_data = map_text_to_fields(sample_front_text, front_fields)
print("\nEXTRACTED FRONT DATA:")
for field, value in front_data.items():
    print(f"{field}: {value}")

# Process back side
print("\nTESTING BACK SIDE EXTRACTION:")
back_fields = formats['bulgarian_id']['back']
success, back_data = map_text_to_fields(sample_back_text, back_fields)
print("\nEXTRACTED BACK DATA:")
for field, value in back_data.items():
    print(f"{field}: {value}")

# Combine and save to file
all_data = {
    "document_type": "bulgarian_id",
    "front": front_data,
    "back": back_data
}

with open('extracted_test_id_data.json', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=4, ensure_ascii=False)

print("\nExtracted test data saved to 'extracted_test_id_data.json'") 