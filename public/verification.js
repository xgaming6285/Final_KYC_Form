// Verification Page JavaScript
let currentStep = 1;
let uploadedFiles = {
    idFront: null,
    idBack: null,
    incomeDoc: null,
    facePhoto: null
};
let currentCameraTarget = null;
let faceStream = null;
let translationManager = null;

// Initialize translation system
document.addEventListener('DOMContentLoaded', async function() {
    // Initialize translation manager
    translationManager = new TranslationManager();
    await translationManager.init();
    
    // Set language to English by default
    if (translationManager.getCurrentLanguage() !== 'en') {
        await translationManager.changeLanguage('en');
    }
    
    // Mobile Navigation Toggle
    const hamburger = document.getElementById('hamburger');
    const navMenu = document.getElementById('nav-menu');
    
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', function() {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });

        // Close mobile menu when clicking on a link
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                hamburger.classList.remove('active');
                navMenu.classList.remove('active');
            });
        });
    }
    
    // Setup file upload areas
    setupFileUpload('id-front', 'id-front-preview');
    setupFileUpload('id-back', 'id-back-preview');
    setupFileUpload('income-doc', 'income-preview');
    setupFileUpload('face-upload', 'face-preview');
    
    // Setup drag and drop
    setupDragAndDrop();
});

// Helper function to get translated text
function t(key, defaultValue = '') {
    return translationManager ? translationManager.t(key, defaultValue) : defaultValue;
}

// Step Navigation
function nextStep(stepNumber) {
    if (validateCurrentStep()) {
        // Mark current step as completed
        const currentStepItem = document.querySelector(`.step-item[data-step="${currentStep}"]`);
        if (currentStepItem) {
            currentStepItem.classList.remove('active');
            currentStepItem.classList.add('completed');
        }
        
        // Show next step
        document.getElementById(`step-${currentStep}`).classList.remove('active');
        document.getElementById(`step-${stepNumber}`).classList.add('active');
        
        // Update step indicator
        const nextStepItem = document.querySelector(`.step-item[data-step="${stepNumber}"]`);
        if (nextStepItem) {
            nextStepItem.classList.add('active');
        }
        
        currentStep = stepNumber;
    }
}

function prevStep(stepNumber) {
    // Mark current step as inactive
    const currentStepItem = document.querySelector(`.step-item[data-step="${currentStep}"]`);
    if (currentStepItem) {
        currentStepItem.classList.remove('active');
    }
    
    // Show previous step
    document.getElementById(`step-${currentStep}`).classList.remove('active');
    document.getElementById(`step-${stepNumber}`).classList.add('active');
    
    // Update step indicator
    const prevStepItem = document.querySelector(`.step-item[data-step="${stepNumber}"]`);
    if (prevStepItem) {
        prevStepItem.classList.remove('completed');
        prevStepItem.classList.add('active');
    }
    
    currentStep = stepNumber;
}

// Validation
function validateCurrentStep() {
    switch (currentStep) {
        case 1:
            return validatePersonalInfo();
        case 2:
            return validateDocuments();
        case 3:
            return validateFaceVerification();
        default:
            return true;
    }
}

function validatePersonalInfo() {
    const form = document.getElementById('personal-info-form');
    const formData = new FormData(form);
    
    // Basic validation
    for (let [key, value] of formData.entries()) {
        if (!value.trim()) {
            alert(t('verification.validation.fill_required', 'Please fill in all required fields.'));
            return false;
        }
    }
    
    // EGN validation (basic)
    const egn = formData.get('egn');
    if (egn && egn.length !== 10) {
        alert(t('verification.validation.egn_invalid', 'Personal ID Number must be 10 digits.'));
        return false;
    }
    
    return true;
}

function validateDocuments() {
    if (!uploadedFiles.idFront || !uploadedFiles.idBack) {
        alert(t('verification.validation.id_cards_required', 'Please upload photos of both sides of your ID card.'));
        return false;
    }
    
    if (!uploadedFiles.incomeDoc) {
        alert(t('verification.validation.income_doc_required', 'Please upload an income document.'));
        return false;
    }
    
    return true;
}

function validateFaceVerification() {
    if (!uploadedFiles.facePhoto) {
        alert(t('verification.validation.face_photo_required', 'Please take a selfie for face verification.'));
        return false;
    }
    
    return true;
}

// File Upload Handling
function setupFileUpload(inputId, previewId) {
    const input = document.getElementById(inputId);
    const uploadArea = document.getElementById(inputId.replace('-', '-') + '-upload') || input.parentElement;
    const preview = document.getElementById(previewId);
    
    if (uploadArea && uploadArea !== input.parentElement) {
        uploadArea.addEventListener('click', () => {
            input.click();
        });
    }
    
    input.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            handleFileUpload(file, inputId, preview);
        }
    });
}

function handleFileUpload(file, inputId, preview) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (inputId === 'income-doc') {
        validTypes.push('application/pdf');
    }
    
    if (!validTypes.includes(file.type)) {
        alert(t('verification.validation.file_invalid', 'Please select a valid file (JPG, PNG, WEBP' + (inputId === 'income-doc' ? ', PDF' : '') + ').'));
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert(t('verification.validation.file_too_large', 'File is too large. Maximum size: 10MB.'));
        return;
    }
    
    // Store file
    const fileKey = inputId.replace('-', '');
    uploadedFiles[fileKey] = file;
    
    // Show preview
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `<img src="${e.target.result}" alt="File preview">`;
        };
        reader.readAsDataURL(file);
    } else {
        preview.innerHTML = `<p><i class="fas fa-file-pdf"></i> ${file.name}</p>`;
    }
}

function setupDragAndDrop() {
    const uploadAreas = document.querySelectorAll('.upload-area');
    
    uploadAreas.forEach(area => {
        area.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.add('dragover');
        });
        
        area.addEventListener('dragleave', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.remove('dragover');
        });
        
        area.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const input = this.querySelector('input[type="file"]');
                if (input) {
                    const inputId = input.id;
                    const previewId = inputId + '-preview';
                    const preview = document.getElementById(previewId);
                    handleFileUpload(files[0], inputId, preview);
                }
            }
        });
    });
}

// Camera Functionality
function openCamera(target) {
    currentCameraTarget = target;
    const cameraModal = document.getElementById('camera-modal');
    cameraModal.classList.add('active');
    
    startCamera();
}

function startCamera() {
    const video = document.getElementById('camera-video');
    
    navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: 'environment',
            width: { ideal: 1920 },
            height: { ideal: 1080 }
        } 
    })
    .then(function(stream) {
        video.srcObject = stream;
        video.play();
    })
    .catch(function(error) {
        console.error('Error accessing camera:', error);
        alert(t('verification.validation.camera_error', 'Cannot access camera. Please use the file upload option.'));
        closeCameraModal();
    });
}

function captureDocument() {
    const video = document.getElementById('camera-video');
    const canvas = document.getElementById('camera-canvas');
    const context = canvas.getContext('2d');
    
    // Set canvas dimensions to video dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to blob
    canvas.toBlob(function(blob) {
        // Create a file-like object
        const file = new File([blob], `document-${Date.now()}.jpg`, { type: 'image/jpeg' });
        
        // Handle the captured image
        const previewId = currentCameraTarget + '-preview';
        const preview = document.getElementById(previewId);
        handleFileUpload(file, currentCameraTarget, preview);
        
        closeCameraModal();
    }, 'image/jpeg', 0.9);
}

function closeCameraModal() {
    const cameraModal = document.getElementById('camera-modal');
    const video = document.getElementById('camera-video');
    
    // Stop camera stream
    if (video.srcObject) {
        const stream = video.srcObject;
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
    }
    
    cameraModal.classList.remove('active');
    currentCameraTarget = null;
}

// Face Verification
function startFaceCapture() {
    const video = document.getElementById('face-video');
    const placeholder = document.getElementById('face-placeholder');
    const captureBtn = document.querySelector('.capture-btn');
    
    navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: 'user',
            width: { ideal: 640 },
            height: { ideal: 480 }
        } 
    })
    .then(function(stream) {
        faceStream = stream;
        video.srcObject = stream;
        video.style.display = 'block';
        placeholder.style.display = 'none';
        captureBtn.style.display = 'inline-flex';
        video.play();
    })
    .catch(function(error) {
        console.error('Error accessing camera:', error);
        alert(t('verification.validation.camera_error', 'Cannot access camera. Please use the file upload option.'));
    });
}

function capturePhoto() {
    const video = document.getElementById('face-video');
    const canvas = document.getElementById('face-canvas');
    const context = canvas.getContext('2d');
    const preview = document.getElementById('face-preview');
    
    // Set canvas dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Stop video stream
    if (faceStream) {
        faceStream.getTracks().forEach(track => track.stop());
        faceStream = null;
    }
    
    video.style.display = 'none';
    document.getElementById('face-placeholder').style.display = 'block';
    document.querySelector('.capture-btn').style.display = 'none';
    
    // Convert to blob and store
    canvas.toBlob(function(blob) {
        const file = new File([blob], `face-${Date.now()}.jpg`, { type: 'image/jpeg' });
        uploadedFiles.facePhoto = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `<img src="${e.target.result}" alt="Face verification selfie">`;
        };
        reader.readAsDataURL(file);
    }, 'image/jpeg', 0.9);
}

function uploadFaceImage() {
    document.getElementById('face-upload').click();
}

// Submit Application
function submitApplication() {
    if (!validateFaceVerification()) {
        return;
    }
    
    // Show loading state
    const submitBtn = document.querySelector('.submit-btn');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${t('verification.uploading', 'Submitting...')}`;
    submitBtn.disabled = true;
    
    // Simulate application submission
    setTimeout(() => {
        // Hide current step
        document.getElementById('step-3').classList.remove('active');
        
        // Show success step
        document.getElementById('step-success').style.display = 'block';
        
        // Update step indicator - mark all as completed
        document.querySelectorAll('.step-item').forEach(item => {
            item.classList.remove('active');
            item.classList.add('completed');
        });
        
        // Reset button
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
        
        // In a real application, you would send the data to your server here
        console.log('Application submitted with files:', uploadedFiles);
        
        // Stop all streams
        stopAllStreams();
    }, 3000);
}

// Stop all camera streams
function stopAllStreams() {
    // Stop face verification stream
    if (faceStream) {
        faceStream.getTracks().forEach(track => track.stop());
        faceStream = null;
    }
    
    // Stop camera modal stream
    const cameraVideo = document.getElementById('camera-video');
    if (cameraVideo && cameraVideo.srcObject) {
        const stream = cameraVideo.srcObject;
        stream.getTracks().forEach(track => track.stop());
        cameraVideo.srcObject = null;
    }
} 