/**
 * Video Recording Manager
 * Handles session-wide video recording for KYC form
 * Records from both front and back cameras if available
 */

class VideoRecordingManager {
    constructor() {
        this.isRecording = false;
        this.mediaRecorders = {};
        this.videoChunks = {};
        this.streams = {};
        this.sessionId = null;
        this.recordingStartTime = null;
        
        // Video recording constraints
        this.videoConstraints = {
            front: {
                video: {
                    facingMode: 'user',
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: true // Include audio for better session documentation
            },
            back: {
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                },
                audio: true
            }
        };
        
        console.log('📹 Video Recording Manager initialized');
    }
    
    /**
     * Initialize video recording session
     */
    async initializeSession(sessionId) {
        this.sessionId = sessionId;
        
        try {
            // Notify server about video recording session start
            const response = await fetch('/start_video_recording', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ session_id: sessionId })
            });
            
            if (response.ok) {
                console.log(`✅ Video recording session initialized: ${sessionId}`);
                return true;
            } else {
                console.error('❌ Failed to initialize video recording session');
                return false;
            }
        } catch (error) {
            console.error('❌ Error initializing video recording session:', error);
            return false;
        }
    }
    
    /**
     * Start recording from available cameras
     */
    async startRecording() {
        if (this.isRecording) {
            console.log('⚠️ Recording already in progress');
            return;
        }
        
        this.recordingStartTime = Date.now();
        console.log('🎬 Starting video recording...');
        
        // Try to start recording from available cameras
        // Skip front camera if we're on face verification page to avoid conflicts
        const isFaceVerificationPage = window.location.pathname.includes('face_verification');
        const cameraTypes = isFaceVerificationPage ? ['back'] : ['front', 'back'];
        
        console.log(`🎥 Starting recording for cameras: ${cameraTypes.join(', ')} ${isFaceVerificationPage ? '(front camera reserved for face verification)' : ''}`);
        
        for (const cameraType of cameraTypes) {
            try {
                await this.startCameraRecording(cameraType);
            } catch (error) {
                console.warn(`⚠️ Could not start ${cameraType} camera recording:`, error.message);
                // Continue with other cameras even if one fails
            }
        }
        
        // Check if at least one camera started recording
        const activeRecordings = Object.keys(this.mediaRecorders).length;
        if (activeRecordings > 0) {
            this.isRecording = true;
            console.log(`✅ Video recording started with ${activeRecordings} camera(s)`);
            this.showRecordingIndicator();
        } else {
            console.error('❌ Failed to start recording from any camera');
            throw new Error('Could not access any cameras for recording');
        }
    }
    
    /**
     * Start recording from a specific camera
     */
    async startCameraRecording(cameraType) {
        try {
            // Get media stream
            const stream = await navigator.mediaDevices.getUserMedia(this.videoConstraints[cameraType]);
            this.streams[cameraType] = stream;
            
            // Check if MediaRecorder is supported
            if (!MediaRecorder.isTypeSupported('video/mp4')) {
                // Fallback to webm if mp4 not supported
                if (!MediaRecorder.isTypeSupported('video/webm')) {
                    throw new Error('Video recording not supported in this browser');
                }
            }
            
            // Create MediaRecorder with MP4 preference
            let mimeType = 'video/mp4';
            let fileExtension = 'mp4';
            
            if (!MediaRecorder.isTypeSupported('video/mp4')) {
                mimeType = 'video/webm;codecs=vp9';
                fileExtension = 'webm';
            }
            
            const mediaRecorder = new MediaRecorder(stream, {
                mimeType: mimeType,
                videoBitsPerSecond: cameraType === 'front' ? 1000000 : 2500000 // Lower bitrate for front camera
            });
            
            this.videoChunks[cameraType] = [];
            this.videoExtension = fileExtension;
            
            // Set up event handlers
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.videoChunks[cameraType].push(event.data);
                }
            };
            
            mediaRecorder.onstop = () => {
                console.log(`📹 ${cameraType} camera recording stopped`);
                // Only upload final video when entire session stops, not individual cameras
                console.log(`⏸️ ${cameraType} camera stopped - final upload will happen when session ends`);
            };
            
            mediaRecorder.onerror = (event) => {
                console.error(`❌ MediaRecorder error for ${cameraType} camera:`, event.error);
            };
            
            // Start recording
            mediaRecorder.start(1000); // Collect data every second
            this.mediaRecorders[cameraType] = mediaRecorder;
            
            console.log(`📹 ${cameraType} camera recording started`);
            
        } catch (error) {
            // If we can't access a camera, don't fail completely
            console.warn(`⚠️ Could not access ${cameraType} camera for recording:`, error.message);
            
            // Special handling for camera conflicts
            if (error.name === 'NotAllowedError') {
                console.warn(`Camera permission denied for ${cameraType} camera`);
            } else if (error.name === 'NotReadableError' || error.name === 'AbortError') {
                console.warn(`Camera already in use or hardware error for ${cameraType} camera`);
            }
            
            throw error;
        }
    }
    
    /**
     * Stop all video recordings
     */
    stopRecording() {
        if (!this.isRecording) {
            console.log('⚠️ No active recording to stop');
            return;
        }
        
        console.log('🛑 Stopping video recording...');
        
        // Stop all media recorders
        Object.keys(this.mediaRecorders).forEach(cameraType => {
            const mediaRecorder = this.mediaRecorders[cameraType];
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
        });
        
        // Stop all streams
        Object.keys(this.streams).forEach(cameraType => {
            const stream = this.streams[cameraType];
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        });
        
        this.isRecording = false;
        this.hideRecordingIndicator();
        
        const recordingDuration = (Date.now() - this.recordingStartTime) / 1000;
        console.log(`✅ Recording stopped after ${recordingDuration.toFixed(1)} seconds`);
        
        // Final video upload is now handled by "Complete KYC" button
        console.log('ℹ️ Video recording stopped - final upload handled by Complete KYC button');
    }
    
    /**
     * Upload recorded video to server
     */
    async uploadVideo(cameraType) {
        if (!this.videoChunks[cameraType] || this.videoChunks[cameraType].length === 0) {
            console.warn(`⚠️ No video data to upload for ${cameraType} camera`);
            return;
        }
        
        try {
            // Create video blob with correct MIME type
            const mimeType = this.videoExtension === 'mp4' ? 'video/mp4' : 'video/webm';
            const videoBlob = new Blob(this.videoChunks[cameraType], { type: mimeType });
            
            console.log(`📤 Uploading ${cameraType} camera video (${(videoBlob.size / 1024 / 1024).toFixed(2)} MB)...`);
            
            // Create form data
            const formData = new FormData();
            formData.append('video', videoBlob, `session_${cameraType}_camera.${this.videoExtension}`);
            formData.append('session_id', this.sessionId);
            formData.append('camera_type', `${cameraType}_camera`);
            
            // Upload to server
            const response = await fetch('/upload_video', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                console.log(`✅ ${cameraType} camera video uploaded successfully: ${result.s3_key}`);
            } else {
                console.error(`❌ Failed to upload ${cameraType} camera video:`, result.error);
            }
            
        } catch (error) {
            console.error(`❌ Error uploading ${cameraType} camera video:`, error);
        }
    }
    
    /**
     * Show recording indicator
     */
    showRecordingIndicator() {
        // Remove existing indicator if present
        this.hideRecordingIndicator();
        
        // Create recording indicator
        const indicator = document.createElement('div');
        indicator.id = 'recording-indicator';
        indicator.innerHTML = `
            <div style="
                position: fixed;
                top: 20px;
                right: 20px;
                background: rgba(255, 0, 0, 0.9);
                color: white;
                padding: 10px 15px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: bold;
                z-index: 10000;
                display: flex;
                align-items: center;
                gap: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            ">
                <div style="
                    width: 12px;
                    height: 12px;
                    background: white;
                    border-radius: 50%;
                    animation: pulse 1s infinite;
                "></div>
                Recording Session
            </div>
            <style>
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
            </style>
        `;
        
        document.body.appendChild(indicator);
    }
    
    /**
     * Hide recording indicator
     */
    hideRecordingIndicator() {
        const indicator = document.getElementById('recording-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    /**
     * Get recording status
     */
    getStatus() {
        return {
            isRecording: this.isRecording,
            sessionId: this.sessionId,
            activeCameras: Object.keys(this.mediaRecorders),
            recordingDuration: this.recordingStartTime ? (Date.now() - this.recordingStartTime) / 1000 : 0
        };
    }
    
    /**
     * Clean up resources
     */
    cleanup() {
        this.stopRecording();
        this.mediaRecorders = {};
        this.videoChunks = {};
        this.streams = {};
        console.log('🧹 Video recording manager cleaned up');
    }

    /**
     * Get available camera stream for sharing (useful for face verification)
     */
    getSharedCameraStream(cameraType = 'front') {
        if (this.streams[cameraType]) {
            return this.streams[cameraType];
        }
        return null;
    }
    
    /**
     * Check if specific camera is being used for recording
     */
    isCameraRecording(cameraType) {
        return this.mediaRecorders[cameraType] && this.mediaRecorders[cameraType].state === 'recording';
    }

    /**
     * Upload the final consolidated video to S3 when recording session ends
     */
    async uploadFinalVideoToS3() {
        if (!this.sessionId || !this.videoChunks) {
            console.log('⚠️ No session data or video chunks to upload');
            return;
        }

        console.log('☁️ Uploading final consolidated video to S3...');

        // Find the camera type with the most video data (likely front camera)
        let bestCameraType = null;
        let maxSize = 0;

        Object.keys(this.videoChunks).forEach(cameraType => {
            if (this.videoChunks[cameraType] && this.videoChunks[cameraType].length > 0) {
                const totalSize = this.videoChunks[cameraType].reduce((sum, chunk) => sum + chunk.size, 0);
                if (totalSize > maxSize) {
                    maxSize = totalSize;
                    bestCameraType = cameraType;
                }
            }
        });

        if (!bestCameraType || maxSize === 0) {
            console.log('⚠️ No video data found to upload to S3');
            return;
        }

        try {
            // Create final video blob from the best camera
            const mimeType = this.videoExtension === 'mp4' ? 'video/mp4' : 'video/webm';
            const finalVideoBlob = new Blob(this.videoChunks[bestCameraType], { type: mimeType });

            console.log(`📤 Uploading final ${bestCameraType} camera video to S3 (${(finalVideoBlob.size / 1024 / 1024).toFixed(2)} MB)...`);

            // Create form data for final upload
            const formData = new FormData();
            formData.append('video', finalVideoBlob, `final_session_${this.sessionId}.${this.videoExtension}`);
            formData.append('session_id', this.sessionId);
            formData.append('camera_type', 'final_session');

            // Upload final video to S3
            const response = await fetch('/upload_video', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                console.log(`✅ Final session video uploaded to S3: ${result.s3_key}`);
                
                // Clean up local video chunks after successful S3 upload
                this.videoChunks = {};
                console.log('🧹 Local video chunks cleaned up after S3 upload');
                
            } else {
                console.error(`❌ Failed to upload final video to S3:`, result.error);
            }

        } catch (error) {
            console.error(`❌ Error uploading final video to S3:`, error);
        }
    }
}

// Global video recording manager instance
window.videoRecordingManager = new VideoRecordingManager();

// Auto-cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.videoRecordingManager) {
        window.videoRecordingManager.stopRecording();
    }
});

console.log('📹 Video Recording Manager module loaded'); 