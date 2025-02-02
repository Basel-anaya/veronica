// Configuration
const CONFIG = {
    FRAMES_REQUIRED: 8,
    MIN_FRAME_INTERVAL: 150,
    MAX_RETRIES: 3,
    RETRY_DELAY: 1000,
    SESSION_TIMEOUT: 10000,
    DEBUG: window.location.hostname === 'localhost' || 
           window.location.hostname === '127.0.0.1' ||
           window.location.hostname === '0.0.0.0',
    MIN_LIVENESS_CONFIDENCE: 0.8,
    BACKOFF_INITIAL: 1000,
    BACKOFF_MAX: 5000,
    RATE_LIMIT: {
        MAX_REQUESTS: 10,
        INTERVAL: 5000,
        REQUESTS: []
    },
    IMAGE_QUALITY: 0.85,
    MAX_IMAGE_DIMENSION: 720,
    MAX_SESSION_ATTEMPTS: 3,
    TOKEN_EXPIRY: 30000,
    SECURITY: {
        REQUIRED_HTTPS: window.location.hostname !== 'localhost' && 
                       window.location.hostname !== '127.0.0.1' &&
                       window.location.hostname !== '0.0.0.0',
        MIN_BROWSER_VERSION: {
            chrome: 60,
            firefox: 60,
            safari: 12,
            edge: 79
        },
        TOKEN_REFRESH_THRESHOLD: 25000,
        DEVELOPMENT_MODE: window.location.hostname === 'localhost' || 
                         window.location.hostname === '127.0.0.1' ||
                         window.location.hostname === '0.0.0.0'
    },
    // Get device pixel ratio and screen dimensions
    devicePixelRatio: window.devicePixelRatio || 1,
    screenWidth: window.innerWidth * (window.devicePixelRatio || 1),
    screenHeight: window.innerHeight * (window.devicePixelRatio || 1),
    ERROR_MESSAGES: {
        CAMERA_DENIED: 'Camera access is required. Please allow access and refresh.',
        SESSION_EXPIRED: 'Session expired. Please try again.',
        NETWORK_ERROR: 'Connection lost. Please check your internet and try again.',
        GENERIC_ERROR: 'Something went wrong. Please try again.',
        FACE_POSITION: 'Please position your face within the oval.',
        PROCESSING: 'Processing your verification...',
        SERVER_ERROR: 'Service temporarily unavailable. Please try again shortly.',
        BROWSER_NOT_SUPPORTED: 'Your browser version is not supported. Please update.',
        HTTPS_REQUIRED: 'Secure connection required. Please use HTTPS.',
        TOKEN_ERROR: 'Authentication token expired. Please refresh.',
        NOT_FOUND: 'Service endpoint not found. Please check configuration.'
    },
    API_BASE_URL: (() => {
        const hostname = window.location.hostname;
        const protocol = window.location.protocol;
        
        // Development environment
        if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '0.0.0.0') {
            return `${protocol}//${hostname}:5000`;
        }
        
        // Production environment
        return `${protocol}//${hostname}`;
    })()
};

// Authentication states
const AUTH_STATE = {
    WAITING: 'waiting',
    ANALYZING: 'analyzing',
    AUTHENTICATED: 'authenticated',
    ERROR: 'error'
};

// Enhanced debug logging
const debug = {
    log: (message, data = null) => {
        if (!CONFIG.DEBUG) return;
        const timestamp = new Date().toISOString();
        console.log(`[Debug ${timestamp}] ${message}`, data || '');
    },
    error: (message, error) => {
        if (!CONFIG.DEBUG) return;
        console.error(`[Error] ${message}`, error);
    }
};

// Global state management
const state = {
    isRunning: false,
    isProcessing: false,
    currentState: AUTH_STATE.WAITING,
    framesCollected: 0,
    stream: null,
    lastProcessedTime: 0,
    sessionTimer: null,
    lastUIUpdate: 0,
    ovalState: {
        current: 'no-face',
        transitioning: false,
        lastUpdate: 0
    },
    sessionId: null,
    retryCount: 0,
    backoffDelay: CONFIG.BACKOFF_INITIAL,
    sessionStartTime: 0,
    lastFrameId: null,
    collectedFrames: new Set(),  // Store unique frame tokens
    sessionTokens: {
        validation: null,
        lastUpdate: 0,
        csrf: null
    },
    errors: {
        count: 0,
        lastError: null,
        lastErrorTime: 0
    },
    performance: {
        startTime: 0,
        frameProcessingTimes: [],
        averageProcessingTime: 0
    }
};

// DOM Elements
const elements = {
    video: null,
    canvas: null,
    ctx: null,
    statusDisplay: null,
    statusHint: null,
    ovalBorder: null
};

// Initialize DOM elements
async function initializeElements() {
    try {
        elements.video = document.getElementById('video');
        elements.canvas = document.getElementById('canvas');
        elements.statusDisplay = document.getElementById('status');
        elements.statusHint = document.getElementById('statusHint');
        elements.ovalBorder = document.querySelector('.oval-border');

        // Verify all elements exist
        const missingElements = Object.entries(elements)
            .filter(([key, value]) => !value && key !== 'ctx')
            .map(([key]) => key);

        if (missingElements.length > 0) {
            throw new Error(`Missing required elements: ${missingElements.join(', ')}`);
        }

        // Initialize canvas context
        elements.ctx = elements.canvas.getContext('2d');
        if (!elements.ctx) {
            throw new Error('Failed to get canvas context');
        }

        debug.log('All elements initialized successfully');
        return true;
    } catch (error) {
        debug.error('Element initialization failed:', error);
        showError(`Initialization Error: ${error.message}. Please refresh the page.`);
        return false;
    }
}

// Show error message
function showError(message) {
    const errorMessage = document.createElement('div');
    errorMessage.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(255, 0, 0, 0.8);
        color: white;
        padding: 20px;
        border-radius: 5px;
        text-align: center;
        z-index: 9999;
    `;
    errorMessage.textContent = message;
    document.body.appendChild(errorMessage);
}

// Add watermark and license display
function addWatermarkAndLicense() {
    // Create watermark
    const watermark = document.createElement('div');
    watermark.className = 'watermark';
    watermark.innerHTML = `
        <div class="watermark-text">
            <span>Veronica Face Authentication</span>
            <span class="version">v2.5.0</span>
        </div>
    `;
    document.body.appendChild(watermark);

    // Create license info
    const license = document.createElement('div');
    license.className = 'license-info';
    license.innerHTML = `
        <div class="license-text">
            <span>© ${new Date().getFullYear()} Barzan DIG</span>
            <span class="separator">|</span>
            <span>All Rights Reserved</span>
            <span class="separator">|</span>
            <a href="#" onclick="showLicenseDetails(event)">License Info</a>
        </div>
    `;
    document.body.appendChild(license);

    // Add styles
    const style = document.createElement('style');
    style.textContent = `
        .watermark {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            opacity: 0.7;
            pointer-events: none;
            transition: opacity 0.3s ease;
        }

        .watermark-text {
            font-family: Arial, sans-serif;
            color: rgba(255, 255, 255, 0.8);
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
            font-size: 14px;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }

        .watermark .version {
            font-size: 12px;
            opacity: 0.8;
        }

        .license-info {
            position: fixed;
            bottom: 10px;
            left: 0;
            right: 0;
            text-align: center;
            z-index: 1000;
            padding: 5px;
            background: rgba(0, 0, 0, 0.5);
        }

        .license-text {
            font-family: Arial, sans-serif;
            color: rgba(255, 255, 255, 0.8);
            font-size: 12px;
            line-height: 1.5;
        }

        .license-text .separator {
            margin: 0 10px;
            opacity: 0.5;
        }

        .license-text a {
            color: rgba(255, 255, 255, 0.8);
            text-decoration: none;
            cursor: pointer;
        }

        .license-text a:hover {
            text-decoration: underline;
            color: white;
        }

        .license-modal {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0, 0, 0, 0.9);
            padding: 20px;
            border-radius: 8px;
            z-index: 2000;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            color: white;
            font-family: Arial, sans-serif;
        }

        .license-modal h2 {
            margin-top: 0;
            color: #2196F3;
        }

        .license-modal .close-button {
            position: absolute;
            top: 10px;
            right: 10px;
            background: none;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            padding: 5px;
        }

        .license-modal .close-button:hover {
            color: #2196F3;
        }
    `;
    document.head.appendChild(style);
}

// Show license details modal
function showLicenseDetails(event) {
    event.preventDefault();
    
    const modal = document.createElement('div');
    modal.className = 'license-modal';
    modal.innerHTML = `
        <button class="close-button" onclick="this.parentElement.remove()">×</button>
        <h2>License Information</h2>
        <div class="license-content">
            <p><strong>Veronica Face Authentication System</strong></p>
            <p>Version 2.5.0</p>
            <p>© ${new Date().getFullYear()} Barzan DIG. All Rights Reserved.</p>
            <br>
            <p>This software is proprietary and confidential. Unauthorized copying, transfer, or use of this software, 
            via any medium, is strictly prohibited without the express written permission of Barzan DIG.</p>
            <br>
            <p>The software is provided "as is", without warranty of any kind, express or implied, including but not limited to 
            the warranties of merchantability, fitness for a particular purpose and noninfringement.</p>
        </div>
    `;
    document.body.appendChild(modal);
}

// Initialize application
async function initializeApp() {
    // Remove any existing debug panel
    const existingDebugPanel = document.getElementById('debugPanel');
    if (existingDebugPanel) {
        existingDebugPanel.remove();
    }
    
    if (await initializeElements()) {
        addWatermarkAndLicense();  // Add watermark and license before starting camera
        await startCamera();
    }
}

// Start initialization when DOM is ready
document.addEventListener('DOMContentLoaded', initializeApp);

// Resource management
const ResourceManager = {
    cleanupResources: () => {
        // Stop video stream
        if (state.stream) {
            state.stream.getTracks().forEach(track => {
                track.stop();
                state.stream.removeTrack(track);
            });
            state.stream = null;
        }

        // Clear canvas
        if (elements.ctx) {
            elements.ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
        }

        // Clear video source
        if (elements.video) {
            elements.video.srcObject = null;
        }

        // Clear timers
        if (state.sessionTimer) {
            clearTimeout(state.sessionTimer);
            state.sessionTimer = null;
        }

        // Clear performance marks
        if (CONFIG.DEBUG) {
            performance.clearMarks();
            performance.clearMeasures();
        }

        // Reset arrays to free memory
        CONFIG.RATE_LIMIT.REQUESTS = [];
        Performance.metrics.frameProcessingTimes = [];
        Performance.metrics.networkLatency = [];
    },

    releaseMemory: () => {
        // Clear any large objects
        state.collectedFrames.clear();
        
        // Clear any cached images or data
        if (elements.canvas) {
            const tempCtx = elements.canvas.getContext('2d');
            tempCtx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
        }
        
        // Run garbage collection hint
        if (window.gc) {
            try {
                window.gc();
            } catch (e) {
                console.warn('Manual GC not available');
            }
        }
    }
};

// Enhanced cleanup on page unload
window.addEventListener('beforeunload', () => {
    ResourceManager.cleanupResources();
    ResourceManager.releaseMemory();
});

// Enhanced cleanup on visibility change
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Pause processing when tab is not visible
        state.isRunning = false;
        ResourceManager.cleanupResources();
    } else {
        // Resume processing when tab becomes visible
        if (state.currentState !== AUTH_STATE.AUTHENTICATED) {
            initializeApp();
        }
    }
});

// Enhanced error handling with retry logic
const ErrorHandler = {
    maxRetries: CONFIG.MAX_RETRIES,
    retryDelay: CONFIG.RETRY_DELAY,
    
    handle: async function(error, context = '') {
        debug.error(`Error in ${context}:`, error);
        
        // Track error in state
        state.errors.count++;
        state.errors.lastError = error;
        state.errors.lastErrorTime = Date.now();

        // Check for critical errors that should stop the process
        if (ErrorHandler.isCriticalError(error)) {
            await ErrorHandler.handleCriticalError(error);
            return;
        }

        // Network errors
        if (error instanceof TypeError && error.message.includes('network')) {
            await updateUI(AUTH_STATE.ERROR, CONFIG.ERROR_MESSAGES.NETWORK_ERROR);
            ErrorHandler.scheduleRetry();
            return;
        }

        // Camera errors
        if (error.name === 'NotAllowedError' || error.name === 'NotFoundError') {
            await updateUI(AUTH_STATE.ERROR, CONFIG.ERROR_MESSAGES.CAMERA_DENIED);
            return;
        }

        // Rate limit errors
        if (error.message.includes('429')) {
            const waitTime = state.backoffDelay;
            await updateUI(
                AUTH_STATE.WAITING,
                'Processing, please wait...',
                `Retrying in ${Math.round(waitTime/1000)} seconds`
            );
            return;
        }

        // Server errors
        if (error.message.includes('500')) {
            await updateUI(AUTH_STATE.ERROR, CONFIG.ERROR_MESSAGES.SERVER_ERROR);
            ErrorHandler.scheduleRetry();
            return;
        }

        // Generic errors
        await updateUI(AUTH_STATE.ERROR, CONFIG.ERROR_MESSAGES.GENERIC_ERROR);
    },

    isCriticalError: function(error) {
        return (
            error.name === 'SecurityError' ||
            error.message.includes('SSL') ||
            error.message.includes('security') ||
            state.errors.count >= CONFIG.MAX_SESSION_ATTEMPTS
        );
    },

    handleCriticalError: async function(error) {
        state.isRunning = false;
        ResourceManager.cleanupResources();
        await updateUI(AUTH_STATE.ERROR, 'Critical error occurred. Please refresh the page.');
    },

    scheduleRetry: function() {
        if (state.retryCount < CONFIG.MAX_RETRIES) {
            state.retryCount++;
            setTimeout(() => {
                if (state.isRunning) {
                    processFrame();
                }
            }, CONFIG.RETRY_DELAY * state.retryCount);
        } else {
            handleSessionTimeout();
        }
    }
};

// Enhanced security checks
const SecurityChecks = {
    validateEnvironment: function() {
        // Only enforce HTTPS in production environments
        if (CONFIG.SECURITY.REQUIRED_HTTPS && window.location.protocol !== 'https:') {
            if (window.location.hostname !== 'localhost' && 
                window.location.hostname !== '127.0.0.1' && 
                window.location.hostname !== '0.0.0.0') {
                throw new Error('Application must run in a secure context (HTTPS) in production');
            }
        }

        // Check for required APIs
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Required media APIs not available');
        }

        // Check browser support
        const browser = SecurityChecks.detectBrowser();
        if (!SecurityChecks.isValidBrowser(browser)) {
            throw new Error('Unsupported browser version');
        }

        // Check for WebAssembly support (if needed for face detection)
        if (typeof WebAssembly !== 'object') {
            throw new Error('WebAssembly not supported');
        }

        return true;
    },

    detectBrowser: () => {
        const ua = navigator.userAgent;
        let browser = {
            name: 'unknown',
            version: 0
        };

        if (ua.includes('Chrome/')) {
            browser.name = 'chrome';
            browser.version = parseInt(ua.split('Chrome/')[1]);
        } else if (ua.includes('Firefox/')) {
            browser.name = 'firefox';
            browser.version = parseInt(ua.split('Firefox/')[1]);
        } else if (ua.includes('Safari/') && !ua.includes('Chrome')) {
            browser.name = 'safari';
            browser.version = parseInt(ua.split('Version/')[1]);
        } else if (ua.includes('Edg/')) {
            browser.name = 'edge';
            browser.version = parseInt(ua.split('Edg/')[1]);
        }

        return browser;
    },

    isValidBrowser: (browser) => {
        const minVersion = CONFIG.SECURITY.MIN_BROWSER_VERSION[browser.name];
        return minVersion && browser.version >= minVersion;
    },

    validateResponse: (response) => {
        if (!response || typeof response !== 'object') return false;
        
        // In development mode, be more lenient
        if (CONFIG.SECURITY.DEVELOPMENT_MODE) {
            // Only check if it's an object with success field
            return typeof response.success === 'boolean';
        }
        
        // Required fields
        const requiredFields = ['success', 'face_detected', 'message'];
        if (!requiredFields.every(field => field in response)) return false;

        // Type checks
        if (typeof response.success !== 'boolean') return false;
        if (typeof response.face_detected !== 'boolean') return false;
        if (typeof response.message !== 'string') return false;

        // Validation token format (if present)
        if (response.validation_token && !/^[a-f0-9]{64}$/i.test(response.validation_token)) {
            return false;
        }

        return true;
    },

    sanitizeInput: (input) => {
        if (typeof input !== 'string') return '';
        return input.replace(/[<>]/g, ''); // Basic XSS prevention
    }
};

// Start session timeout
function startSessionTimeout() {
    if (state.sessionTimer) {
        clearTimeout(state.sessionTimer);
    }
    
    state.sessionTimer = setTimeout(async () => {
        // Don't expire if authenticated
        if (state.currentState === AUTH_STATE.AUTHENTICATED) {
            return;
        }

        debug.log('Session timeout reached');
        await handleSessionTimeout();
    }, CONFIG.SESSION_TIMEOUT);
}

// Start camera with production settings
async function startCamera() {
    try {
        if (!SecurityChecks.validateEnvironment()) {
            throw new Error('Security checks failed');
        }

        if (state.isRunning) return;
        
        // Reset all state
        resetState();
        
        const constraints = {
            video: {
                width: { ideal: CONFIG.screenWidth },
                height: { ideal: CONFIG.screenHeight },
                facingMode: 'user',
                frameRate: { ideal: 30, max: 60 },
                resizeMode: 'crop-and-scale'
            }
        };
        
        state.stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        const videoTrack = state.stream.getVideoTracks()[0];
        const settings = videoTrack.getSettings();
        
        // Set video element dimensions to match screen
        elements.video.style.width = '100%';
        elements.video.style.height = '100%';
        elements.video.srcObject = state.stream;
        elements.video.setAttribute('playsinline', true);
        
        await new Promise((resolve) => {
            elements.video.onloadedmetadata = () => {
                elements.video.play();
                resolve();
            };
        });
        
        // Wait for video to stabilize
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Initialize session only if not in development mode
        if (!CONFIG.SECURITY.DEVELOPMENT_MODE) {
            try {
                const response = await fetch(`${CONFIG.API_BASE_URL}/api/start-session`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }
                });

                if (!response.ok) {
                    throw new Error(`Failed to start session: ${response.status}`);
                }

                const sessionData = await response.json();
                state.sessionId = sessionData.session_id;
                state.sessionTokens.validation = sessionData.validation_token;
                state.sessionTokens.csrf = sessionData.csrf_token;
                state.sessionStartTime = Date.now();
                
                debug.log('Session initialized:', state.sessionId);
            } catch (error) {
                debug.error('Session initialization failed:', error);
                throw new Error('Failed to initialize session with server');
            }
        } else {
            // In development mode, use dummy session data
            debug.log('Development mode: Using dummy session');
            state.sessionId = 'dev_session';
            state.sessionTokens.validation = 'dev_token';
            state.sessionTokens.csrf = 'dev_csrf';
            state.sessionStartTime = Date.now();
        }
        
        // Start session
        state.isRunning = true;
        
        // Start session timeout
        startSessionTimeout();
        
        await updateUI(AUTH_STATE.WAITING, 'Position your face in the oval');
        processFrame();
        
        // Start performance monitoring
        state.performance.startTime = performance.now();
        
    } catch (error) {
        await ErrorHandler.handle(error, 'startCamera');
    }
}

// Generate unique session ID
function generateSessionId() {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 15);
    const sessionId = `session_${timestamp}_${random}`;
    debug.log('Generated new session ID:', sessionId);
    return sessionId;
}

// Enhanced token management
function updateSessionToken(newToken) {
    if (!newToken) {
        debug.log('Invalid token received: token is empty');
        return false;
    }
    
    const now = Date.now();
    
    // First token initialization
    if (!state.sessionTokens.validation) {
        debug.log('Initializing first token');
        state.sessionTokens.validation = newToken;
        state.sessionTokens.lastUpdate = now;
        scheduleTokenRefresh();
        return true;
    }
    
    // Don't update if token hasn't changed
    if (state.sessionTokens.validation === newToken) {
        debug.log('Skipping identical token');
        return false;
    }
    
    // Update token if it's different
    debug.log('Updating to new token');
    state.sessionTokens.validation = newToken;
    state.sessionTokens.lastUpdate = now;
    scheduleTokenRefresh();
    return true;
}

// Schedule token refresh
function scheduleTokenRefresh() {
    if (state.tokenRefreshTimer) {
        clearTimeout(state.tokenRefreshTimer);
    }
    
    state.tokenRefreshTimer = setTimeout(() => {
        if (state.isRunning && !state.isProcessing) {
            debug.log('Initiating token refresh');
            processFrame();
        }
    }, CONFIG.SECURITY.TOKEN_REFRESH_THRESHOLD);
}

// Handle session timeout
async function handleSessionTimeout() {
    debug.log('Handling session timeout');
    state.isRunning = false;
    state.isProcessing = false;
    
    ResourceManager.cleanupResources();
    
    await updateUI(AUTH_STATE.ERROR, 'Session expired', 'Please refresh to try again');
    showTimeoutOverlay();
}

// Core UI update function
async function updateUI(newState, message = '', hint = '') {
    if (state.currentState === AUTH_STATE.AUTHENTICATED) {
        return;
    }

    const now = Date.now();
    if (now - state.lastUIUpdate < 300) {
        return;
    }
    state.lastUIUpdate = now;

    try {
        // Remove previous state
        elements.ovalBorder.classList.remove('analyzing', 'success', 'no-face', 'detected', 'error');
        elements.statusDisplay.classList.remove('analyzing', 'success', 'error', 'no-face');

        // Update state-specific UI
        let newOvalState = '';
        switch (newState) {
            case AUTH_STATE.ANALYZING:
                newOvalState = 'analyzing';
                elements.ovalBorder.classList.add('analyzing');
                elements.statusDisplay.classList.add('analyzing');
                break;

            case AUTH_STATE.AUTHENTICATED:
                newOvalState = 'success';
                elements.ovalBorder.classList.add('success');
                elements.statusDisplay.classList.add('success');
                break;

            case AUTH_STATE.ERROR:
                newOvalState = 'error';
                elements.ovalBorder.classList.add('error');
                elements.statusDisplay.classList.add('error');
                break;

            case AUTH_STATE.WAITING:
            default:
                newOvalState = message.includes('Processing') ? 'detected' : 'no-face';
                elements.ovalBorder.classList.add(newOvalState);
                break;
        }

        // Always ensure status is visible
        elements.statusDisplay.classList.add('visible');

        // Update state and messages
        state.ovalState.current = newOvalState;
        state.ovalState.lastUpdate = now;
        
        // Update messages with position check
        if (message) {
            // Replace any directional messages with generic positioning message
            if (message.includes('Move your face') || 
                message.includes('move up') || 
                message.includes('move down') || 
                message.includes('move left') || 
                message.includes('move right') ||
                message.includes('closer') ||
                message.includes('back')) {
                message = CONFIG.ERROR_MESSAGES.FACE_POSITION;
            }
            elements.statusDisplay.textContent = message;
        }
        if (hint) {
            elements.statusHint.textContent = hint;
        }
        
        state.currentState = newState;
    } catch (error) {
        console.error('Error updating UI:', error);
    }
}

// Calculate head size ratio relative to oval
function calculateHeadSizeRatio(ovalData, faceRect) {
    if (!faceRect) return 0;
    
    // Calculate the area of the oval and face rectangle
    const ovalArea = Math.PI * (ovalData.width / 2) * (ovalData.height / 2);
    const faceArea = faceRect.width * faceRect.height;
    
    // Calculate ratio of face area to oval area
    const ratio = faceArea / ovalArea;
    debug.log('Head size ratio:', ratio);
    
    return ratio;
}

// Dynamic face position checking with improved size calculation
function checkFacePosition(ovalData, faceRect) {
    if (!faceRect) {
        debug.log('No face rectangle data available');
        return { isValid: false, message: CONFIG.ERROR_MESSAGES.FACE_POSITION };
    }
    
    // Calculate centers
    const ovalCenterX = ovalData.x + ovalData.width / 2;
    const ovalCenterY = ovalData.y + ovalData.height / 2;
    const faceCenterX = faceRect.x + faceRect.width / 2;
    const faceCenterY = faceRect.y + faceRect.height / 2;
    
    // Calculate distances as percentages of oval dimensions
    const xDistancePercent = Math.abs(faceCenterX - ovalCenterX) / (ovalData.width / 2) * 100;
    const yDistancePercent = Math.abs(faceCenterY - ovalCenterY) / (ovalData.height / 2) * 100;
    
    // Calculate face-to-oval ratio using area comparison
    const ovalArea = Math.PI * (ovalData.width / 2) * (ovalData.height / 2);
    const faceArea = faceRect.width * faceRect.height;
    const areaRatio = faceArea / ovalArea;
    
    // Dynamic thresholds based on frame height
    const frameHeight = ovalData.frame_height;
    const scaleFactor = frameHeight / 480; // Base scale on standard 480p height
    
    // More lenient thresholds with dynamic scaling
    const maxDistancePercent = 60 * scaleFactor;  // Increased from 50 to 60
    const minAreaRatio = 0.35 * scaleFactor;      // Decreased from 0.4 to 0.35
    const maxAreaRatio = 0.95 * scaleFactor;      // Increased from 0.9 to 0.95
    
    debug.log('Face position metrics:', {
        xDistancePercent,
        yDistancePercent,
        areaRatio,
        scaleFactor,
        thresholds: {
            maxDistance: maxDistancePercent,
            minArea: minAreaRatio,
            maxArea: maxAreaRatio
        }
    });
    
    // Position check
    if (xDistancePercent > maxDistancePercent || yDistancePercent > maxDistancePercent) {
        return { 
            isValid: false, 
            message: CONFIG.ERROR_MESSAGES.FACE_POSITION,
            details: {
                reason: 'position',
                xOffset: xDistancePercent,
                yOffset: yDistancePercent
            }
        };
    }
    
    // Size check
    if (areaRatio < minAreaRatio) {
        return { 
            isValid: false, 
            message: 'Please move closer to the camera',
            details: {
                reason: 'too_small',
                ratio: areaRatio
            }
        };
    }
    
    if (areaRatio > maxAreaRatio) {
        return { 
            isValid: false, 
            message: 'Please move back from the camera',
            details: {
                reason: 'too_large',
                ratio: areaRatio
            }
        };
    }
    
    return { 
        isValid: true, 
        message: 'Face position is good',
        details: {
            areaRatio,
            xOffset: xDistancePercent,
            yOffset: yDistancePercent
        }
    };
}

// Enhanced capture frame with optimized image capture
async function captureFrame() {
    try {
        // Wait for video to be ready
        if (!elements.video || !elements.video.readyState === elements.video.HAVE_ENOUGH_DATA) {
            debug.log('Video not ready yet, waiting...');
            await new Promise(resolve => setTimeout(resolve, 100));
            return null;
        }

        // Double check video dimensions
        if (!elements.video.videoWidth || !elements.video.videoHeight) {
            debug.log('Video dimensions not available yet, retrying...');
            await new Promise(resolve => setTimeout(resolve, 100));
            return null;
        }
        
        // Calculate optimal dimensions based on video aspect ratio
        const videoAspectRatio = elements.video.videoWidth / elements.video.videoHeight;
        const maxDimension = CONFIG.MAX_IMAGE_DIMENSION;
        
        let targetWidth, targetHeight;
        if (elements.video.videoWidth > elements.video.videoHeight) {
            targetWidth = Math.min(elements.video.videoWidth, maxDimension);
            targetHeight = Math.round(targetWidth / videoAspectRatio);
        } else {
            targetHeight = Math.min(elements.video.videoHeight, maxDimension);
            targetWidth = Math.round(targetHeight * videoAspectRatio);
        }
        
        // Set canvas dimensions
        elements.canvas.width = targetWidth;
        elements.canvas.height = targetHeight;
        
        // Draw video frame to canvas
        elements.ctx.drawImage(elements.video, 0, 0, targetWidth, targetHeight);
        
        // Get oval guide dimensions
        const ovalGuide = elements.ovalBorder;
        if (!ovalGuide) {
            throw new Error('Oval guide element not found');
        }

        const videoRect = elements.video.getBoundingClientRect();
        const ovalRect = ovalGuide.getBoundingClientRect();
        
        // Calculate scale factors
        const scaleX = targetWidth / videoRect.width;
        const scaleY = targetHeight / videoRect.height;
        
        // Calculate oval dimensions in video coordinates
        const ovalData = {
            x: Math.round((ovalRect.left - videoRect.left) * scaleX),
            y: Math.round((ovalRect.top - videoRect.top) * scaleY),
            width: Math.round(ovalRect.width * scaleX),
            height: Math.round(ovalRect.height * scaleY),
            frame_height: targetHeight  // Add frame height to oval data
        };
        
        // Get image data with specified quality
        const imageData = elements.canvas.toDataURL('image/jpeg', CONFIG.IMAGE_QUALITY);
        
        return { imageData, ovalData };
    } catch (error) {
        debug.error('Error in captureFrame:', error);
        return null;
    }
}

// Show progress overlay
function showProgressOverlay() {
    // Remove any existing overlay first
    const existingOverlay = document.querySelector('.progress-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }

    const overlay = document.createElement('div');
    overlay.className = 'progress-overlay';
    
    const content = document.createElement('div');
    content.className = 'progress-content';
    
    const title = document.createElement('h2');
    title.textContent = 'Verifying Identity';
    title.style.marginBottom = '15px';
    
    const message = document.createElement('div');
    message.textContent = 'Collecting secure frames...';
    
    // Create dots container
    const dotsContainer = document.createElement('div');
    dotsContainer.className = 'progress-dots';
    
    // Add dots for each required frame
    for (let i = 0; i < CONFIG.FRAMES_REQUIRED; i++) {
        const dot = document.createElement('div');
        dot.className = 'progress-dot';
        dotsContainer.appendChild(dot);
    }
    
    const progressText = document.createElement('div');
    progressText.className = 'progress-text';
    progressText.textContent = '0%';
    
    content.appendChild(title);
    content.appendChild(message);
    content.appendChild(dotsContainer);
    content.appendChild(progressText);
    overlay.appendChild(content);
    
    document.body.appendChild(overlay);
    
    // Force a reflow to ensure the overlay is visible
    overlay.offsetHeight;
}

// Update progress overlay
function updateProgressOverlay(framesCollected) {
    const progressOverlay = document.querySelector('.progress-overlay');
    if (!progressOverlay) {
        showProgressOverlay();
        return;
    }
    
    const percentage = Math.round((framesCollected / CONFIG.FRAMES_REQUIRED) * 100);
    const progressText = progressOverlay.querySelector('.progress-text');
    
    // Update dots
    const dots = progressOverlay.querySelectorAll('.progress-dot');
    dots.forEach((dot, index) => {
        if (index < framesCollected) {
            dot.classList.add('active');
        }
    });
    
    if (progressText) {
        progressText.textContent = `${percentage}%`;
    }
    
    const message = progressOverlay.querySelector('.progress-content div:nth-child(2)');
    if (message) {
        message.textContent = `Collecting secure frames (${framesCollected}/${CONFIG.FRAMES_REQUIRED})`;
    }
}

// Reset state helper
function resetState() {
    state.framesCollected = 0;
    state.lastFrameId = null;
    state.collectedFrames.clear();
    state.sessionTokens.validation = null;
    state.sessionTokens.lastUpdate = 0;
    state.errors.count = 0;
    state.errors.lastError = null;
    state.performance.frameProcessingTimes = [];
}

// Show progress dots
function showProgressDots() {
    // Remove any existing dots
    const existingDots = document.querySelector('.progress-dots');
    if (existingDots) {
        existingDots.remove();
    }

    const dotsContainer = document.createElement('div');
    dotsContainer.className = 'progress-dots';
    dotsContainer.style.cssText = `
        position: absolute;
        bottom: -40px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        justify-content: center;
        gap: 12px;
        z-index: 100;
    `;
    
    // Add dots for each required frame
    for (let i = 0; i < CONFIG.FRAMES_REQUIRED; i++) {
        const dot = document.createElement('div');
        dot.className = 'progress-dot';
        dot.style.cssText = `
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.3);
            transition: all 0.3s ease-in-out;
            position: relative;
            display: block;
            box-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
        `;
        dotsContainer.appendChild(dot);
    }
    
    // Add dots container to the oval container
    const ovalContainer = document.querySelector('.oval-border');
    if (ovalContainer) {
        ovalContainer.appendChild(dotsContainer);
    }
}

// Update progress dots
function updateProgressDots(framesCollected) {
    let dotsContainer = document.querySelector('.progress-dots');
    if (!dotsContainer) {
        showProgressDots();
        dotsContainer = document.querySelector('.progress-dots');
    }
    
    if (dotsContainer) {
        const dots = dotsContainer.querySelectorAll('.progress-dot');
        dots.forEach((dot, index) => {
            if (index < framesCollected) {
                dot.style.backgroundColor = 'var(--primary-color)';
                dot.style.transform = 'scale(1.1)';
                dot.style.boxShadow = '0 0 10px rgba(33, 150, 243, 0.5)';
                
                // Add pulsing animation
                if (!dot.style.animation) {
                    dot.style.animation = 'pulse-border 1.5s infinite';
                }
            } else {
                dot.style.backgroundColor = 'rgba(255, 255, 255, 0.3)';
                dot.style.transform = 'scale(1)';
                dot.style.boxShadow = '0 0 5px rgba(0, 0, 0, 0.3)';
                dot.style.animation = 'none';
            }
        });
    }
}

// Add keyframes for pulse animation
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse-border {
        0% {
            transform: scale(1);
            opacity: 1;
        }
        50% {
            transform: scale(1.3);
            opacity: 0.5;
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }
`;
document.head.appendChild(style);

// Performance monitoring
const Performance = {
    metrics: {
        frameProcessingTimes: [],
        networkLatency: [],
        startupTime: 0,
        lastFrameTime: 0
    },

    mark: (name) => {
        if (CONFIG.DEBUG) {
            performance.mark(name);
        }
    },

    measure: (name, startMark, endMark) => {
        if (CONFIG.DEBUG) {
            try {
                performance.measure(name, startMark, endMark);
                const measure = performance.getEntriesByName(name).pop();
                return measure ? measure.duration : 0;
            } catch (e) {
                return 0;
            }
        }
        return 0;
    },

    trackFrameProcessing: (duration) => {
        Performance.metrics.frameProcessingTimes.push(duration);
        if (Performance.metrics.frameProcessingTimes.length > 50) {
            Performance.metrics.frameProcessingTimes.shift();
        }
    },

    trackNetworkLatency: (duration) => {
        Performance.metrics.networkLatency.push(duration);
        if (Performance.metrics.networkLatency.length > 50) {
            Performance.metrics.networkLatency.shift();
        }
    },

    getAverageFrameTime: () => {
        if (Performance.metrics.frameProcessingTimes.length === 0) return 0;
        const sum = Performance.metrics.frameProcessingTimes.reduce((a, b) => a + b, 0);
        return sum / Performance.metrics.frameProcessingTimes.length;
    },

    getAverageNetworkLatency: () => {
        if (Performance.metrics.networkLatency.length === 0) return 0;
        const sum = Performance.metrics.networkLatency.reduce((a, b) => a + b, 0);
        return sum / Performance.metrics.networkLatency.length;
    }
};

// Enhanced process frame with performance monitoring
async function processFrame() {
    if (!state.isRunning || state.isProcessing) return;
    
    Performance.mark('frameStart');
    const now = Date.now();
    
    // Rate limiting check
    const recentRequests = CONFIG.RATE_LIMIT.REQUESTS.filter(
        time => now - time < CONFIG.RATE_LIMIT.INTERVAL
    );
    CONFIG.RATE_LIMIT.REQUESTS = recentRequests;
    
    if (recentRequests.length >= CONFIG.RATE_LIMIT.MAX_REQUESTS) {
        setTimeout(() => requestAnimationFrame(processFrame), 100);
        return;
    }
    
    // Quick interval check with dynamic adjustment
    const timeSinceLastProcess = now - state.lastProcessedTime;
    const currentFrameInterval = Math.max(
        CONFIG.MIN_FRAME_INTERVAL,
        Performance.getAverageFrameTime()
    );
    
    if (timeSinceLastProcess < currentFrameInterval) {
        requestAnimationFrame(processFrame);
        return;
    }
    
    state.isProcessing = true;
    CONFIG.RATE_LIMIT.REQUESTS.push(now);
    
    try {
        Performance.mark('captureStart');
        const frameData = await captureFrame();
        
        // If frame capture failed, retry
        if (!frameData) {
            state.isProcessing = false;
            setTimeout(() => requestAnimationFrame(processFrame), 100);
            return;
        }

        const { imageData, ovalData } = frameData;
        Performance.measure('captureTime', 'captureStart', 'frameStart');
        
        Performance.mark('networkStart');
        
        // Debug log headers before sending
        debug.log('Sending request with tokens:', {
            sessionId: state.sessionId,
            validationToken: state.sessionTokens.validation,
            csrfToken: state.sessionTokens.csrf,
            developmentMode: CONFIG.SECURITY.DEVELOPMENT_MODE
        });

        const headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };

        // Only add authentication headers if not in development mode
        if (!CONFIG.SECURITY.DEVELOPMENT_MODE) {
            headers['session-token'] = state.sessionTokens.validation;
            headers['csrf-token'] = state.sessionTokens.csrf;
            headers['X-Session-ID'] = state.sessionId;
        }

        const response = await fetch(`${CONFIG.API_BASE_URL}/api/authenticate`, {
            method: 'POST',
            headers: headers,
            credentials: CONFIG.SECURITY.DEVELOPMENT_MODE ? 'omit' : 'include',
            body: JSON.stringify({ 
                image: imageData, 
                oval_guide: ovalData,
                session_id: CONFIG.SECURITY.DEVELOPMENT_MODE ? 'dev_session' : state.sessionId
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            debug.error('Authentication error:', {
                status: response.status,
                statusText: response.statusText,
                error: errorText
            });

            if (response.status === 404) {
                debug.error('API endpoint not found:', `${CONFIG.API_BASE_URL}/api/authenticate`);
                throw new Error(CONFIG.ERROR_MESSAGES.NOT_FOUND);
            }
            if (response.status === 401) {
                debug.error('Session invalid, restarting session...');
                await startCamera();  // Restart the session
                return;
            }
            if (response.status === 429) {
                state.backoffDelay = Math.min(state.backoffDelay * 1.5, CONFIG.BACKOFF_MAX);
                setTimeout(() => {
                    state.isProcessing = false;
                    requestAnimationFrame(processFrame);
                }, state.backoffDelay);
                return;
            }
            throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
        }

        const result = await response.json();
        if (!SecurityChecks.validateResponse(result)) {
            throw new Error('Invalid response format');
        }

        // Process result and update UI
        if (result.success && result.face_detected) {
            if (result.message === 'Face verified successfully' && result.is_live) {
                // Always increment frame count for successful verifications
                state.framesCollected++;
                updateProgressDots(state.framesCollected);
                debug.log(`Collected frame ${state.framesCollected}/${CONFIG.FRAMES_REQUIRED}`);
                
                // Store the validation token if present
                if (result.validation_token) {
                    state.sessionTokens.validation = result.validation_token;
                    state.sessionTokens.lastUpdate = now;
                }
                
                if (state.framesCollected >= CONFIG.FRAMES_REQUIRED) {
                    if (result.recognized_name) {
                        await handleSuccessfulAuthentication(result.recognized_name);
                        return;
                    }
                }
                
                await updateUI(AUTH_STATE.ANALYZING, 'Verifying...', 'Keep still');
            } else {
                await updateUI(AUTH_STATE.WAITING, result.message || 'Adjust position');
            }
        } else {
            await updateUI(AUTH_STATE.WAITING, 'Position face in oval');
        }

    } catch (error) {
        await ErrorHandler.handle(error, 'processFrame');
    } finally {
        Performance.mark('frameEnd');
        const frameTime = Performance.measure('frameTime', 'frameStart', 'frameEnd');
        Performance.trackFrameProcessing(frameTime);
        
        state.lastProcessedTime = now;
        state.isProcessing = false;
        
        if (state.isRunning && state.currentState !== AUTH_STATE.AUTHENTICATED) {
            requestAnimationFrame(processFrame);
        }
    }
}

// Handle successful authentication
async function handleSuccessfulAuthentication(name) {
    if (!name) {
        console.error('Cannot handle authentication: name is undefined');
        return;
    }
    
    // Clear session timeout
    if (state.sessionTimer) {
        clearTimeout(state.sessionTimer);
        state.sessionTimer = null;
    }
    
    state.isRunning = false;
    state.isProcessing = false;
    state.currentState = AUTH_STATE.AUTHENTICATED;
    
    const firstName = SecurityChecks.sanitizeInput(name.split(' ')[0]);
    await updateUI(AUTH_STATE.AUTHENTICATED, `Welcome, ${firstName}!`, 'Authentication successful');
    
    ResourceManager.cleanupResources();
    
    // Show success overlay with sanitized input
    showSuccessOverlay(firstName);
    
    // Notify parent application with sanitized data
    notifyParentApplication({
        status: 'success',
        name: SecurityChecks.sanitizeInput(name),
        message: 'Authentication successful',
        timestamp: Date.now(),
        sessionId: state.sessionId
    });
}

// Show success overlay
function showSuccessOverlay(firstName) {
    // Remove any existing overlays first
    const existingOverlay = document.querySelector('.success-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }

    const successOverlay = document.createElement('div');
    successOverlay.className = 'success-overlay active';
    successOverlay.innerHTML = `
        <div class="success-content">
            <h2 style="margin-bottom: 20px;">Authentication Successful</h2>
            <div class="success-icon">✓</div>
            <p style="font-size: 1.3rem; margin: 20px 0;">Welcome, ${SecurityChecks.sanitizeInput(firstName)}!</p>
            <button onclick="window.location.reload()" class="action-button">
                Start New Session
            </button>
        </div>
    `;
    document.body.appendChild(successOverlay);
    
    // Add success animation
    const successIcon = successOverlay.querySelector('.success-icon');
    if (successIcon) {
        setTimeout(() => successIcon.classList.add('animate'), 100);
    }
}

// Show timeout overlay
function showTimeoutOverlay() {
    const timeoutOverlay = document.createElement('div');
    timeoutOverlay.className = 'timeout-overlay active';
    timeoutOverlay.innerHTML = `
        <div class="timeout-content">
            <h2>Session Expired</h2>
            <p>Your authentication session has timed out.</p>
            <button onclick="window.location.reload()" class="action-button">
                Try Again
            </button>
        </div>
    `;
    document.body.appendChild(timeoutOverlay);
}

// Notify parent application
function notifyParentApplication(data) {
    if (window.flutter_inappwebview) {
        window.flutter_inappwebview.callHandler('authenticationComplete', {
            ...data,
            timestamp: Date.now()
        });
    } else if (window.parent && window.parent !== window) {
        // For iframe integration
        window.parent.postMessage({
            type: 'authenticationComplete',
            data: {
                ...data,
                timestamp: Date.now()
            }
        }, '*');
    }
    debug.log('Notified parent application:', data);
}