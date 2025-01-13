// Camera elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const status = document.getElementById('status');
const processedImageContainer = document.getElementById('processedImageContainer');

// Registration elements
const personNameInput = document.getElementById('personName');
const idCardInput = document.getElementById('idCard');
const idPreview = document.getElementById('idPreview');
const registerBtn = document.getElementById('registerBtn');
const uploadArea = document.querySelector('.file-upload');
const uploadPlaceholder = document.querySelector('.upload-placeholder');
const previewPlaceholder = document.querySelector('.preview-placeholder');

// Log if elements are found
console.log('Elements found:', {
    video: !!video,
    canvas: !!canvas,
    startBtn: !!startBtn,
    status: !!status,
    personNameInput: !!personNameInput,
    idCardInput: !!idCardInput,
    idPreview: !!idPreview,
    registerBtn: !!registerBtn,
    uploadArea: !!uploadArea
});

let isRunning = false;
let authenticationLoop = null;

// Check if running on localhost
const isLocalhost = window.location.hostname === 'localhost' || 
                   window.location.hostname === '127.0.0.1' ||
                   window.location.hostname === '';

// Check if running on HTTPS
const isSecure = window.location.protocol === 'https:';

// Check camera availability
async function checkCameraAvailability() {
    try {
        // Check if mediaDevices is supported
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            if (!isSecure && !isLocalhost) {
                throw new Error('Camera access requires HTTPS or localhost');
            } else {
                throw new Error('Camera access not supported in this browser');
            }
        }

        // Check if camera is accessible
        const response = await fetch('/camera-status', {
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        });
        const result = await response.json();
        if (result.status === 'error') {
            throw new Error(result.message);
        }

        status.textContent = 'Camera is available. Click "Start Camera" to begin.';
        status.className = 'status success';
        startBtn.disabled = false;
    } catch (err) {
        console.error('Camera check error:', err);
        status.textContent = err.message;
        status.className = 'status error';
        startBtn.disabled = true;
    }
}

// Check camera on page load
checkCameraAvailability();

// Request camera access
startBtn.onclick = async () => {
    if (isRunning) {
        // Stop the camera
        isRunning = false;
        if (authenticationLoop) {
            clearTimeout(authenticationLoop);
            authenticationLoop = null;
        }
        startBtn.textContent = 'Start Camera';
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
        video.srcObject = null;
        status.textContent = 'Camera stopped';
        status.className = 'status';
        processedImageContainer.innerHTML = '';
        return;
    }

    try {
        const constraints = { 
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            }
        };
        
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        await video.play();  // Ensure video starts playing
        isRunning = true;
        startBtn.textContent = 'Stop Camera';
        status.textContent = 'Camera started. Initializing face detection...';
        status.className = 'status success';
        startAuthentication();
    } catch (err) {
        console.error('Error accessing camera:', err);
        let errorMessage = err.message;
        if (!isSecure && !isLocalhost) {
            errorMessage = 'Camera access requires HTTPS or localhost. Please use a secure connection.';
        }
        status.textContent = `Error accessing camera: ${errorMessage}`;
        status.className = 'status error';
    }
};

async function startAuthentication() {
    if (!isRunning) return;

    try {
        // Ensure video is ready
        if (video.readyState !== video.HAVE_ENOUGH_DATA) {
            authenticationLoop = setTimeout(startAuthentication, 100);
            return;
        }

        // Capture frame from video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        
        // Convert to base64
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send to server
        const response = await fetch('/authenticate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify({
                image: imageData
            })
        });
        
        const result = await response.json();
        
        // Update status
        status.textContent = result.message;
        if (result.is_live) {
            status.className = 'status success';
            if (result.recognized_name) {
                status.textContent += `\nWelcome, ${result.recognized_name}!`;
            }
        } else {
            status.className = 'status warning';
        }
        
        // Display processed image if available
        if (result.processed_image) {
            processedImageContainer.innerHTML = `
                <img src="${result.processed_image}" alt="Processed face">
            `;
        }
        
    } catch (err) {
        console.error('Authentication error:', err);
        status.textContent = `Authentication error: ${err.message}`;
        status.className = 'status error';
    }
    
    // Schedule next authentication if still running
    if (isRunning) {
        authenticationLoop = setTimeout(startAuthentication, 100);
    }
}

// File Upload Handling
uploadArea.onclick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    idCardInput.click();
};

// Drag and drop handling
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
    });
});

['dragenter', 'dragover'].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => {
        uploadArea.classList.add('highlight');
    });
});

['dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => {
        uploadArea.classList.remove('highlight');
    });
});

uploadArea.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
        idCardInput.files = e.dataTransfer.files;
        handleFile(file);
    }
});

// When a file is selected
idCardInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
};

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }

    // Validate file size (5MB max)
    if (file.size > 5 * 1024 * 1024) {
        alert('File size should be less than 5MB');
        return;
    }

    console.log('Processing file:', file.name, file.type, file.size);
    
    // Show file preview
    const reader = new FileReader();
    reader.onload = (e) => {
        console.log('File loaded successfully');
        idPreview.src = e.target.result;
        idPreview.style.display = 'block';
        uploadPlaceholder.style.display = 'none';
        previewPlaceholder.style.display = 'block';
        
        // Enable register button if name is entered
        registerBtn.disabled = !personNameInput.value.trim();
    };
    reader.onerror = (error) => {
        console.error('Error reading file:', error);
        alert('Error reading file. Please try again.');
    };
    reader.readAsDataURL(file);
}

// Enable/disable register button based on name input
personNameInput.addEventListener('input', () => {
    registerBtn.disabled = !personNameInput.value.trim() || !idCardInput.files[0];
});

// Handle registration
registerBtn.addEventListener('click', async () => {
    const name = personNameInput.value.trim();
    const file = idCardInput.files[0];
    
    if (!name || !file) {
        alert('Please enter a name and select an ID card image');
        return;
    }
    
    try {
        // Show loading state
        registerBtn.disabled = true;
        registerBtn.classList.add('loading');
        
        // Create FormData
        const formData = new FormData();
        formData.append('name', name);
        formData.append('id_card', file);
        
        console.log('Submitting registration:', { name, file });
        
        // Send to server
        const response = await fetch('/register', {
            method: 'POST',
            body: formData
        });
        
        console.log('Registration response:', response);
        
        if (!response.ok) {
            const result = await response.json();
            throw new Error(result.message || 'Registration failed');
        }
        
        const result = await response.json();
        console.log('Registration result:', result);
        
        // Show success message
        status.textContent = result.message;
        status.className = 'status success';
        
        // Reset form
        personNameInput.value = '';
        idCardInput.value = '';
        idPreview.src = '';
        idPreview.style.display = 'none';
        uploadPlaceholder.style.display = 'block';
        previewPlaceholder.style.display = 'none';
        
    } catch (err) {
        console.error('Registration error:', err);
        status.textContent = err.message;
        status.className = 'status error';
    } finally {
        // Reset button state
        registerBtn.disabled = false;
        registerBtn.classList.remove('loading');
    }
}); 