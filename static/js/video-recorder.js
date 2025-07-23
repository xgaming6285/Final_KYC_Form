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
        this.autoSaveInterval = null;
        this.backupSaveInterval = null;
        
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
        
        // Set up auto-save and backup intervals
        this.setupAutoSave();
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
                    console.log(`📊 Video chunk received for ${cameraType}: ${event.data.size} bytes (total chunks: ${this.videoChunks[cameraType].length})`);
                    
                    // Log total accumulated size periodically
                    if (this.videoChunks[cameraType].length % 10 === 0) {
                        const totalSize = this.videoChunks[cameraType].reduce((sum, chunk) => sum + chunk.size, 0);
                        console.log(`📈 ${cameraType} camera accumulated: ${this.videoChunks[cameraType].length} chunks, ${(totalSize / 1024 / 1024).toFixed(2)} MB total`);
                    }
                } else {
                    console.log(`⚠️ Empty video chunk received for ${cameraType} camera`);
                }
            };
            
            mediaRecorder.onstop = () => {
                console.log(`📹 ${cameraType} camera recording stopped`);
                
                // Always upload final video when stopping (regardless of duration)
                const chunkCount = this.videoChunks[cameraType] ? this.videoChunks[cameraType].length : 0;
                
                if (chunkCount > 0) { 
                    const recordingDuration = this.recordingStartTime ? (Date.now() - this.recordingStartTime) / 1000 : 0;
                    console.log(`📤 Uploading final ${cameraType} camera video (${recordingDuration.toFixed(1)}s, ${chunkCount} chunks)`);
                    setTimeout(() => this.uploadVideo(cameraType), 500); // Small delay to ensure all chunks are ready
                } else {
                    console.log(`⚠️ No video data to upload for ${cameraType} camera`);
                }
            };
            
            mediaRecorder.onstart = () => {
                console.log(`📹 ${cameraType} camera recording started`);
            };
            
            mediaRecorder.onpause = () => {
                console.log(`⏸️ ${cameraType} camera recording paused`);
            };
            
            mediaRecorder.onresume = () => {
                console.log(`▶️ ${cameraType} camera recording resumed`);
            };
            
            mediaRecorder.onerror = (event) => {
                console.error(`❌ MediaRecorder error for ${cameraType} camera:`, event.error);
            };
            
            // Start recording
            mediaRecorder.start(1000); // Collect data every second
            this.mediaRecorders[cameraType] = mediaRecorder;
            
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
        
        // Clear auto-save intervals
        if (this.autoSaveInterval) {
            clearInterval(this.autoSaveInterval);
            this.autoSaveInterval = null;
        }
        if (this.backupSaveInterval) {
            clearInterval(this.backupSaveInterval);
            this.backupSaveInterval = null;
        }
        
        // Save current progress before stopping
        this.saveCurrentVideoProgress();
        
        // Wait a moment for progress saving, then stop
        setTimeout(() => {
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
        }, 2000); // Give time for final progress save
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
            
            // Check if blob has meaningful size (more than just headers)
            if (videoBlob.size < 512) { // Less than 512 bytes is likely empty
                console.warn(`⚠️ Video blob too small (${videoBlob.size} bytes) for ${cameraType} camera - skipping upload`);
                return;
            }
            
            const recordingDuration = this.recordingStartTime ? (Date.now() - this.recordingStartTime) / 1000 : 0;
            console.log(`📤 Uploading ${cameraType} camera video (${(videoBlob.size / 1024 / 1024).toFixed(2)} MB, ${recordingDuration.toFixed(1)}s)...`);
            
            // Create form data
            const formData = new FormData();
            formData.append('video', videoBlob, `session_${cameraType}_camera.${this.videoExtension}`);
            formData.append('session_id', this.sessionId);
            formData.append('camera_type', `${cameraType}_camera`);
            
            // Upload to server with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
            
            const response = await fetch('/upload_video', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`Upload failed with status ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                console.log(`✅ ${cameraType} camera video uploaded successfully: ${result.s3_key}`);
            } else {
                console.error(`❌ Failed to upload ${cameraType} camera video:`, result.error);
            }
            
        } catch (error) {
            if (error.name === 'AbortError') {
                console.error(`❌ Upload timeout for ${cameraType} camera video`);
            } else {
                console.error(`❌ Error uploading ${cameraType} camera video:`, error);
            }
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
        // Save final progress before cleanup
        if (this.isRecording) {
            this.saveCurrentVideoProgress();
        }
        
        // Clear auto-save intervals
        if (this.autoSaveInterval) {
            clearInterval(this.autoSaveInterval);
            this.autoSaveInterval = null;
        }
        
        if (this.backupSaveInterval) {
            clearInterval(this.backupSaveInterval);
            this.backupSaveInterval = null;
        }
        
        // Stop recording and clean up
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
     * Set up automatic video saving and backup
     */
    setupAutoSave() {
        // Auto-save video state every 30 seconds
        this.autoSaveInterval = setInterval(() => {
            if (this.isRecording) {
                this.saveVideoState();
            }
        }, 30000);
        
        // Create local backup every 60 seconds
        this.backupSaveInterval = setInterval(() => {
            if (this.isRecording) {
                this.backupVideoLocally();
            }
        }, 60000);
        
        console.log('⏰ Auto-save intervals set up (state: 30s, backup: 60s)');
    }
    
    /**
     * Save current video state to localStorage
     */
    saveVideoState() {
        if (!this.sessionId || !this.isRecording) return;
        
        try {
            const state = {
                sessionId: this.sessionId,
                recordingStartTime: this.recordingStartTime,
                activeCameras: Object.keys(this.mediaRecorders),
                chunkCounts: {},
                totalSize: {}
            };
            
            // Store chunk counts and sizes for each camera
            Object.keys(this.videoChunks).forEach(cameraType => {
                if (this.videoChunks[cameraType]) {
                    state.chunkCounts[cameraType] = this.videoChunks[cameraType].length;
                    state.totalSize[cameraType] = this.videoChunks[cameraType].reduce((total, chunk) => total + chunk.size, 0);
                }
            });
            
            localStorage.setItem('kyc_video_state', JSON.stringify(state));
            console.log(`💾 Video state saved: ${Object.keys(state.chunkCounts).length} cameras, total chunks: ${Object.values(state.chunkCounts).reduce((a, b) => a + b, 0)}`);
            
        } catch (error) {
            console.warn('⚠️ Failed to save video state:', error);
        }
    }
    
    /**
     * Create local backup of current video chunks
     */
    async backupVideoLocally() {
        if (!this.isRecording || !this.videoChunks) return;
        
        Object.keys(this.videoChunks).forEach(async (cameraType) => {
            if (this.videoChunks[cameraType] && this.videoChunks[cameraType].length > 0) {
                try {
                    const mimeType = this.videoExtension === 'mp4' ? 'video/mp4' : 'video/webm';
                    const videoBlob = new Blob(this.videoChunks[cameraType], { type: mimeType });
                    
                    if (videoBlob.size > 1024) { // Only backup if meaningful size
                        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                        const filename = `kyc_backup_${this.sessionId}_${cameraType}_${timestamp}.${this.videoExtension}`;
                        
                        // Save to server as backup
                        const formData = new FormData();
                        formData.append('video', videoBlob, filename);
                        formData.append('session_id', this.sessionId);
                        formData.append('camera_type', `${cameraType}_backup`);
                        formData.append('is_backup', 'true');
                        
                        fetch('/save_video_backup', {
                            method: 'POST',
                            body: formData
                        }).then(response => {
                            if (response.ok) {
                                console.log(`💾 Local backup saved for ${cameraType} camera (${(videoBlob.size / 1024 / 1024).toFixed(2)} MB) - S3 upload on session end`);
                            }
                        }).catch(error => {
                            console.warn(`⚠️ Backup save failed for ${cameraType}:`, error);
                        });
                    }
                } catch (error) {
                    console.warn(`⚠️ Error creating backup for ${cameraType}:`, error);
                }
            }
        });
    }
    
    /**
     * Setup auto-save functionality
     */
    setupAutoSave() {
        // Auto-save every 30 seconds during recording for better chunk collection
        this.autoSaveInterval = setInterval(async () => {
            if (this.isRecording) {
                console.log('📁 Auto-saving video progress...');
                await this.saveCurrentVideoProgress();
            }
        }, 30000); // 30 seconds - more frequent for better data collection
        
        // Backup save every 15 seconds (more frequent for safety)
        this.backupSaveInterval = setInterval(async () => {
            if (this.isRecording) {
                await this.backupVideoLocally();
            }
        }, 15000); // 15 seconds - very frequent to ensure data is not lost
    }

    /**
     * Save current video progress without stopping recording
     */
    async saveCurrentVideoProgress() {
        if (!this.isRecording) {
            console.log('⚠️ No active recording to save');
            return;
        }
        
        console.log('💾 Saving current video progress...');
        
        // Request data from all active recorders without stopping them
        Object.keys(this.mediaRecorders).forEach(cameraType => {
            const mediaRecorder = this.mediaRecorders[cameraType];
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                // This will trigger ondataavailable event with current data
                mediaRecorder.requestData();
            }
        });
        
        // Save state and create backup after a short delay
        setTimeout(() => {
            this.saveVideoState();
            this.backupVideoLocally();
        }, 1000);
        
        console.log('✅ Video progress save initiated');
    }

    /**
     * Upload the final consolidated video to S3 when recording session ends
     */
    async uploadFinalVideoToS3() {
        console.log('🚀 Starting uploadFinalVideoToS3 process...');
        
        if (!this.sessionId) {
            console.log('⚠️ No session ID - cannot upload final video');
            return { success: false, error: 'No session ID available' };
        }

        if (!this.videoChunks || Object.keys(this.videoChunks).length === 0) {
            console.log('⚠️ No video chunks available for upload');
            return { success: false, error: 'No video chunks available' };
        }

        console.log('☁️ Uploading final consolidated video to S3...');
        console.log('📊 Available video chunks:', Object.keys(this.videoChunks));

        // Find the camera type with the most video data
        let bestCameraType = null;
        let maxSize = 0;
        let totalChunks = 0;

        Object.keys(this.videoChunks).forEach(cameraType => {
            if (this.videoChunks[cameraType] && this.videoChunks[cameraType].length > 0) {
                const chunkCount = this.videoChunks[cameraType].length;
                const totalSize = this.videoChunks[cameraType].reduce((sum, chunk) => sum + chunk.size, 0);
                console.log(`📹 ${cameraType} camera: ${chunkCount} chunks, ${(totalSize / 1024 / 1024).toFixed(2)} MB`);
                
                totalChunks += chunkCount;
                if (totalSize > maxSize) {
                    maxSize = totalSize;
                    bestCameraType = cameraType;
                }
            } else {
                console.log(`📹 ${cameraType} camera: no chunks available`);
            }
        });

        if (!bestCameraType || maxSize === 0) {
            console.log('⚠️ No video data found to upload to S3');
            // Try to find any available chunks even if size is 0
            for (const cameraType of Object.keys(this.videoChunks)) {
                if (this.videoChunks[cameraType] && this.videoChunks[cameraType].length > 0) {
                    bestCameraType = cameraType;
                    console.log(`🔄 Found fallback camera type: ${cameraType}`);
                    break;
                }
            }
            
            if (!bestCameraType) {
                return { success: false, error: 'No video data available for upload' };
            }
        }

        console.log(`🎯 Selected ${bestCameraType} camera for final upload (${totalChunks} total chunks, ${(maxSize / 1024 / 1024).toFixed(2)} MB)`);

        try {
            // Ensure video chunks are properly collected before creating blob
            if (this.isRecording) {
                console.log('🔄 Requesting final data from active recorders...');
                // Request final data from any active recorders
                Object.keys(this.mediaRecorders).forEach(cameraType => {
                    const mediaRecorder = this.mediaRecorders[cameraType];
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        mediaRecorder.requestData();
                    }
                });
                
                // Wait a moment for final chunks to be collected
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                // Recalculate sizes after final data collection
                maxSize = this.videoChunks[bestCameraType] ? 
                    this.videoChunks[bestCameraType].reduce((sum, chunk) => sum + chunk.size, 0) : 0;
                console.log(`📊 Updated size after final data collection: ${(maxSize / 1024 / 1024).toFixed(2)} MB`);
            }

            // Create final video blob from the best camera
            const mimeType = this.videoExtension === 'mp4' ? 'video/mp4' : 'video/webm';
            const finalVideoBlob = new Blob(this.videoChunks[bestCameraType], { type: mimeType });

            console.log(`📤 Creating final video blob: ${(finalVideoBlob.size / 1024 / 1024).toFixed(2)} MB, type: ${mimeType}`);

            if (finalVideoBlob.size < 1024) { // Must be at least 1KB to be meaningful
                console.log(`❌ Final video blob too small (${finalVideoBlob.size} bytes) - likely empty recording`);
                return { success: false, error: 'Final video recording is too small or empty' };
            }

            // Create form data for final upload
            const formData = new FormData();
            const filename = `final_session_${this.sessionId}.${this.videoExtension || 'mp4'}`;
            formData.append('video', finalVideoBlob, filename);
            formData.append('session_id', this.sessionId);
            formData.append('camera_type', 'final_session');

            console.log(`📡 Uploading final video to server: ${filename}`);

            // Upload final video to S3 with timeout
            const uploadPromise = fetch('/upload_video', {
                method: 'POST',
                body: formData
            });

            // Add timeout to upload request
            const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Upload timeout')), 60000) // 60 second timeout
            );

            const response = await Promise.race([uploadPromise, timeoutPromise]);
            const result = await response.json();

            if (response.ok && result.success) {
                console.log(`✅ Final session video uploaded to S3: ${result.s3_key}`);
                
                // Clean up local video chunks after successful S3 upload
                this.videoChunks = {};
                console.log('🧹 Local video chunks cleaned up after S3 upload');
                
                return { 
                    success: true, 
                    s3_key: result.s3_key, 
                    s3_url: result.s3_url,
                    message: `Successfully uploaded ${(finalVideoBlob.size / 1024 / 1024).toFixed(2)} MB video` 
                };
                
            } else {
                console.error(`❌ Failed to upload final video to S3:`, result.error || 'Unknown error');
                return { 
                    success: false, 
                    error: result.error || `HTTP ${response.status}: ${response.statusText}` 
                };
            }

        } catch (error) {
            console.error(`❌ Error uploading final video to S3:`, error);
            return { 
                success: false, 
                error: error.message || 'Upload failed due to network or server error' 
            };
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