"""
Image Quality Validator for ID Cards
Validates that captured ID images meet quality requirements:
- Image is not blurred
- All 4 corners of ID are clearly visible
- Image is clear and properly lit
"""

import cv2
import numpy as np

class ImageQualityValidator:
    def __init__(self):
        pass
    
    def detect_blur(self, image, threshold=50):
        """
        Detect if image is blurred using Laplacian variance method.
        Returns True if image is clear (not blurred), False if blurred.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Higher variance means sharper image
        return laplacian_var > threshold, laplacian_var
    
    def detect_id_card_corners(self, image):
        """
        Detect if all 4 corners of ID card are visible in the image.
        Returns corners if found, None otherwise.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection with more lenient parameters
        edges = cv2.Canny(blur, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (descending)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Look for rectangular contours
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # If our approximated contour has four points, we can assume it's the ID card
            if len(approx) == 4:
                # Check if it's large enough to be an ID card (reduced threshold)
                area = cv2.contourArea(contour)
                if area > 15000:  # Reduced minimum area for ID card
                    return approx, area
        
        return None, 0
    
    def check_lighting(self, image):
        """
        Check if image has adequate lighting.
        Returns True if lighting is good, False if too dark/bright.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate mean brightness
        mean_brightness = np.mean(gray)
        
        # Check if brightness is in acceptable range
        # Too dark: < 50, too bright: > 200
        is_good_lighting = 50 <= mean_brightness <= 200
        
        return is_good_lighting, mean_brightness
    
    def calculate_image_sharpness_score(self, image):
        """
        Calculate overall sharpness score using multiple methods.
        Returns score from 0-100 (higher is better).
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Method 1: Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Method 2: Sobel gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_score = np.mean(sobel_magnitude)
        
        # Normalize scores to 0-100 scale
        laplacian_score = min(100, (laplacian_var / 1000) * 100)
        sobel_norm_score = min(100, (sobel_score / 50) * 100)
        
        # Combined score
        combined_score = (laplacian_score + sobel_norm_score) / 2
        
        return combined_score, {
            'laplacian_variance': laplacian_var,
            'sobel_score': sobel_score,
            'laplacian_normalized': laplacian_score,
            'sobel_normalized': sobel_norm_score
        }
    
    def validate_id_image(self, image):
        """
        Comprehensive validation of ID card image quality.
        Returns validation result with detailed feedback.
        """
        results = {
            'is_valid': False,
            'issues': [],
            'scores': {},
            'corners_detected': False,
            'corners': None,
            'recommendations': []
        }
        
        # 1. Check for blur
        is_clear, blur_score = self.detect_blur(image)
        results['scores']['blur_score'] = float(blur_score)  # Convert to Python float
        
        if not is_clear:
            results['issues'].append('Image appears blurred')
            results['recommendations'].append('Hold the camera steady and ensure proper focus')
        
        # 2. Check for ID card corners
        corners, card_area = self.detect_id_card_corners(image)
        
        # Convert corners to JSON-serializable format
        if corners is not None:
            # Convert numpy array to list of coordinate pairs
            corners_list = []
            for point in corners:
                corners_list.append({
                    'x': int(point[0][0]),
                    'y': int(point[0][1])
                })
            results['corners'] = corners_list
            results['corners_detected'] = True
        else:
            results['corners'] = None
            results['corners_detected'] = False
            
        results['scores']['card_area'] = float(card_area)
        
        if corners is None:
            results['issues'].append('Cannot detect all 4 corners of ID card')
            results['recommendations'].append('Ensure the entire ID card is visible within the frame')
        
        # 3. Check lighting
        is_good_lighting, brightness = self.check_lighting(image)
        results['scores']['brightness'] = float(brightness)  # Convert to Python float
        
        if not is_good_lighting:
            if brightness < 50:
                results['issues'].append('Image is too dark')
                results['recommendations'].append('Improve lighting or move to a brighter area')
            elif brightness > 200:
                results['issues'].append('Image is too bright/overexposed')
                results['recommendations'].append('Reduce lighting or avoid direct sunlight')
        
        # 4. Calculate overall sharpness score
        sharpness_score, sharpness_details = self.calculate_image_sharpness_score(image)
        results['scores']['sharpness'] = float(sharpness_score)  # Convert to Python float
        
        # Convert sharpness details to JSON-serializable format
        results['scores']['sharpness_details'] = {
            'laplacian_variance': float(sharpness_details['laplacian_variance']),
            'sobel_score': float(sharpness_details['sobel_score']),
            'laplacian_normalized': float(sharpness_details['laplacian_normalized']),
            'sobel_normalized': float(sharpness_details['sobel_normalized'])
        }
        
        if sharpness_score < 20:
            results['issues'].append('Image lacks sufficient detail/sharpness')
            results['recommendations'].append('Ensure proper focus and reduce camera shake')
        
        # 5. Overall validation
        results['is_valid'] = bool(
            is_clear and 
            corners is not None and 
            is_good_lighting and 
            sharpness_score >= 20
        )
        
        # 6. Generate quality score (0-100)
        quality_factors = []
        if is_clear:
            quality_factors.append(25)
        if corners is not None:
            quality_factors.append(25)
        if is_good_lighting:
            quality_factors.append(25)
        if sharpness_score >= 20: # Changed from 30 to 20
            quality_factors.append(25)
        
        results['quality_score'] = int(sum(quality_factors))  # Convert to Python int
        
        return results
    
    def get_quality_feedback(self, validation_result):
        """
        Generate user-friendly feedback based on validation results.
        """
        if validation_result['is_valid']:
            return {
                'status': 'success',
                'title': 'Great! ID card image looks good',
                'message': f"Quality score: {int(validation_result['quality_score'])}/100",
                'color': 'green'
            }
        else:
            issues_text = '. '.join(validation_result['issues'])
            recommendations_text = '. '.join(validation_result['recommendations'])
            
            return {
                'status': 'error',
                'title': 'Image quality needs improvement',
                'message': f"Issues: {issues_text}",
                'recommendations': recommendations_text,
                'quality_score': int(validation_result['quality_score']),
                'color': 'red'
            } 