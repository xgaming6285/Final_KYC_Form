# Manual Capture Mode - User Guide

## ✅ **What's New**

The application has been completely redesigned to focus on **manual capture** with **quality validation** instead of auto-capture and text extraction.

### **Key Changes:**

1. **❌ Removed:** Auto-capture functionality
2. **❌ Removed:** Text extraction and OCR processing  
3. **✅ Added:** Manual capture with user control
4. **✅ Added:** Image quality validation (blur, corners, lighting)
5. **✅ Added:** AWS Rekognition face verification
6. **✅ Added:** Simplified 3-step workflow

---

## 🎯 **New Workflow**

### **Step 1: Capture Front Side**
- Position front of ID card in camera frame
- Ensure all 4 corners are visible
- Click "Capture" when ready
- System validates image quality
- Proceed to back side if validation passes

### **Step 2: Capture Back Side**  
- Position back of ID card in camera frame
- Same quality validation process
- Proceed to face verification if validation passes

### **Step 3: Face Verification**
- Take a selfie
- AWS Rekognition compares selfie with front ID photo
- Get instant verification result with similarity score

---

## ✅ **Quality Validation Features**

The system now checks:

### **Blur Detection**
- Uses Laplacian variance to detect blur
- Ensures images are sharp and clear

### **Corner Detection** 
- Verifies all 4 corners of ID card are visible
- Ensures complete ID card capture

### **Lighting Check**
- Validates adequate lighting (not too dark/bright)
- Brightness range: 50-200 (0-255 scale)

### **Sharpness Score**
- Combines multiple sharpness metrics
- Provides 0-100 quality score

---

## 🚀 **How to Use**

### **Option 1: Manual Capture (Recommended)**
```
Visit: http://your-server-ip:5000/simple_capture
```
- Clean, simple interface
- Full manual control
- Step-by-step guidance
- Real-time quality feedback

### **Option 2: Advanced Camera Mode**
```
Visit: http://your-server-ip:5000/camera
```
- Original interface (now with quality validation)
- More features but more complex

### **Option 3: Face Verification Only**
```
Visit: http://your-server-ip:5000/face_verification
```
- If you already have captured ID images

---

## 🔧 **Technical Details**

### **Image Quality Validation**
```python
# Quality factors checked:
- Blur Score: Laplacian variance > 100
- Corner Detection: 4-point contour detection
- Lighting: Mean brightness 50-200
- Sharpness: Combined gradient metrics
```

### **AWS Rekognition Integration**
- **Face Detection:** Finds faces in both images
- **Face Comparison:** Calculates similarity percentage  
- **Threshold:** 85% similarity required for match
- **Security:** Professional-grade verification

### **Session Management**
- Each session gets unique ID
- Links front/back images with face verification
- Temporary storage with automatic cleanup

---

## 📱 **Mobile Optimization**

### **Camera Settings**
- **Facing Mode:** Environment (rear camera)
- **Resolution:** 1280x720 for optimal quality
- **Format:** JPEG with 80% quality

### **UI Features**
- **Responsive Design:** Works on all screen sizes
- **Touch Friendly:** Large buttons and clear feedback
- **Progress Indicators:** Shows current step and completion

---

## 🛡️ **Security & Privacy**

### **AWS Rekognition Benefits**
- **No Local Processing:** Face recognition handled in cloud
- **Enterprise Security:** Bank-grade facial verification  
- **No Permanent Storage:** Images not stored on AWS servers
- **Encrypted Transmission:** All data encrypted in transit

### **Local Image Handling**
- **Temporary Storage:** Images stored locally during session
- **Automatic Cleanup:** Old images automatically removed
- **Quality Only:** No text extraction or data mining

---

## 🔍 **Troubleshooting**

### **Quality Validation Fails**
```
❌ "Image appears blurred"
✅ Hold camera steady, ensure good focus

❌ "Cannot detect all 4 corners" 
✅ Ensure entire ID card visible in frame

❌ "Image is too dark/bright"
✅ Adjust lighting, avoid shadows/glare

❌ "Image lacks sufficient detail"
✅ Move closer, ensure text is readable
```

### **Face Verification Issues**
```
❌ "No matching faces found"
✅ Ensure good lighting for both images
✅ Remove glasses/hats if possible
✅ Try to match angle of ID photo

❌ "AWS Rekognition error" 
✅ Check internet connection
✅ Verify AWS credentials are set up
```

---

## 🎉 **Benefits of Manual Capture**

### **For Users:**
- **Full Control:** Capture when ready
- **Better Quality:** No rushed auto-capture
- **Clear Feedback:** Know exactly what's wrong
- **Faster Process:** No waiting for auto-detection

### **For Administrators:**
- **Higher Success Rate:** Better quality images
- **Reduced Support:** Clear error messages  
- **AWS Integration:** Professional face verification
- **Simplified Code:** Easier to maintain and debug

---

## 🚀 **Getting Started**

1. **Start the server:**
   ```bash
   python web_server.py
   ```

2. **Visit the application:**
   ```
   http://your-ip:5000
   ```

3. **Choose "Manual Capture"**

4. **Follow the 3-step process:**
   - Capture front ID
   - Capture back ID  
   - Take selfie for verification

That's it! The new system is designed to be simple, reliable, and secure. 🎯 