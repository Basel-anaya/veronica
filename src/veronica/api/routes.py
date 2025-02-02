from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends, Cookie, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import base64
import cv2
import numpy as np
import logging
import time
import uuid
from dataclasses import dataclass
from collections import defaultdict
import threading
from veronica.core.face_system import FaceSystem
import random
import os
from dotenv import load_dotenv
from fastapi import Response
import hashlib
import secrets
import hmac
import jwt
from pathlib import Path

# Get the base directory (where src/ is located)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Load environment variables from config/.env
env_path = BASE_DIR / 'config' / '.env'
load_dotenv(env_path)

# Enhanced logging configuration with environment-based settings
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/auth_service.log')
    ]
)
logger = logging.getLogger("auth_service")

# Log the environment loading
logger.info(f"Loading environment from: {env_path}")
logger.info(f"Environment loaded: {os.getenv('ENVIRONMENT', 'Not set')}")

# Environment-specific settings
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
DEBUG = ENVIRONMENT == 'development'
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', '*').split(',')

# Constants with environment-specific values
SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', '30'))  # seconds
MIN_FRAME_INTERVAL = float(os.getenv('MIN_FRAME_INTERVAL', '0.2'))  # seconds
MAX_SESSIONS_PER_IP = int(os.getenv('MAX_SESSIONS_PER_IP', '3'))
RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', '60'))  # seconds
MAX_REQUESTS_PER_WINDOW = int(os.getenv('MAX_REQUESTS_PER_WINDOW', '100'))

# Security settings with better error handling
try:
    # In development mode, use default keys if not provided
    if DEBUG:
        SECRET_KEY = os.getenv('SECRET_KEY', 'development_secret_key_do_not_use_in_production')
        CSRF_KEY = os.getenv('CSRF_KEY', 'development_csrf_key_do_not_use_in_production')
        logger.warning("Using development keys - DO NOT USE IN PRODUCTION!")
    else:
        # Production mode requires proper keys
        SECRET_KEY = os.getenv('SECRET_KEY')
        if not SECRET_KEY:
            # Try to load from secret.key file as fallback
            secret_key_path = BASE_DIR / 'secret.key'
            if secret_key_path.exists():
                SECRET_KEY = secret_key_path.read_text().strip()
                logger.info("Loaded SECRET_KEY from secret.key file")
            else:
                raise RuntimeError("SECRET_KEY must be set in production environment")
        
        CSRF_KEY = os.getenv('CSRF_KEY')
        if not CSRF_KEY:
            # Try to load from csrf.key file as fallback
            csrf_key_path = BASE_DIR / 'csrf.key'
            if csrf_key_path.exists():
                CSRF_KEY = csrf_key_path.read_text().strip()
                logger.info("Loaded CSRF_KEY from csrf.key file")
            else:
                raise RuntimeError("CSRF_KEY must be set in production environment")
    
    logger.info(f"Security keys loaded successfully for {ENVIRONMENT} environment")
except Exception as e:
    logger.error(f"Error loading security keys: {str(e)}")
    raise

# Security settings
ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '30'))

# Session and request models with validation
@dataclass
class SessionMetrics:
    total_sessions: int = 0
    active_sessions: int = 0
    expired_sessions: int = 0
    failed_sessions: int = 0
    avg_session_duration: float = 0.0
    success_rate: float = 0.0

class SessionStats:
    def __init__(self):
        self._metrics = defaultdict(int)
        self._lock = threading.Lock()
    
    def increment(self, metric: str):
        with self._lock:
            self._metrics[metric] += 1
    
    def get_metrics(self) -> dict:
        with self._lock:
            return dict(self._metrics)

class Session:
    def __init__(self, session_id: str, validation_token: str):
        self.id = session_id
        self.validation_token = validation_token
        self.created_at = datetime.now()
        self.last_update = datetime.now()
        self.last_frame_time = datetime.now()
        self.frames_processed = 0
        self.face_detected = False
        self.successful_frames = 0
        self.failed_frames = 0
        self.client_ip = None
        self.user_agent = None
        self.status = "active"
        self.attempts = 0
        self.locked_until = None

    def update(self, frame_success: bool = True):
        self.last_update = datetime.now()
        self.last_frame_time = datetime.now()
        self.frames_processed += 1
        if frame_success:
            self.successful_frames += 1
            self.attempts = 0
        else:
            self.failed_frames += 1
            self.attempts += 1
            if self.attempts >= 3:
                lockout_duration = min(30, 2 ** (self.attempts - 3))
                self.locked_until = datetime.now() + timedelta(seconds=lockout_duration)

    def is_locked(self) -> bool:
        if self.locked_until and datetime.now() < self.locked_until:
            return True
        return False

    def is_expired(self, timeout: int) -> bool:
        return (datetime.now() - self.last_update).total_seconds() > timeout

    def get_duration(self) -> float:
        return (datetime.now() - self.created_at).total_seconds()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "last_update": self.last_update.isoformat(),
            "frames_processed": self.frames_processed,
            "successful_frames": self.successful_frames,
            "failed_frames": self.failed_frames,
            "status": self.status,
            "duration": self.get_duration(),
            "attempts": self.attempts,
            "locked_until": self.locked_until.isoformat() if self.locked_until else None
        }

class SessionManager:
    def __init__(self, cleanup_interval: int = 60):
        self.sessions: Dict[str, Session] = {}
        self.stats = SessionStats()
        self.cleanup_interval = cleanup_interval
        self._lock = threading.Lock()
        self.ip_attempts: Dict[str, List[float]] = defaultdict(list)
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        def cleanup_task():
            while True:
                self.cleanup_expired_sessions()
                self.cleanup_ip_attempts()
                time.sleep(self.cleanup_interval)
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()

    def check_rate_limit(self, client_ip: str) -> bool:
        current_time = time.time()
        self.ip_attempts[client_ip] = [t for t in self.ip_attempts[client_ip] if current_time - t < RATE_LIMIT_WINDOW]
        return len(self.ip_attempts[client_ip]) < MAX_REQUESTS_PER_WINDOW

    def record_attempt(self, client_ip: str):
        self.ip_attempts[client_ip].append(time.time())

    def cleanup_ip_attempts(self):
        current_time = time.time()
        with self._lock:
            for ip in list(self.ip_attempts.keys()):
                self.ip_attempts[ip] = [t for t in self.ip_attempts[ip] if current_time - t < RATE_LIMIT_WINDOW]
                if not self.ip_attempts[ip]:
                    del self.ip_attempts[ip]

    def create_session(self, request: Request, session_id: Optional[str] = None) -> Session:
        client_ip = request.client.host
        if not self.check_rate_limit(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": str(RATE_LIMIT_WINDOW)})
        
        with self._lock:
            if not session_id:
                session_id, validation_token = generate_secure_session_id(client_ip)
            else:
                if not validate_session_id(session_id, client_ip):
                    raise HTTPException(status_code=400, detail="Invalid session ID")
                validation_token = hmac.new(SECRET_KEY.encode(), session_id.encode(), hashlib.sha256).hexdigest()
            
            session = Session(session_id, validation_token)
            session.client_ip = client_ip
            session.user_agent = request.headers.get("user-agent")
            self.sessions[session_id] = session
            self.stats.increment("total_sessions")
            self.stats.increment("active_sessions")
            self.record_attempt(client_ip)
            return session

    def get_session(self, session_id: str, client_ip: str) -> Optional[Session]:
        session = self.sessions.get(session_id)
        if session and validate_session_id(session_id, client_ip):
            if session.is_locked():
                raise HTTPException(status_code=429, detail="Session locked", headers={"Retry-After": str((session.locked_until - datetime.now()).seconds)})
            return session
        return None

    def update_session(self, session_id: str, frame_success: bool = True):
        if session := self.sessions.get(session_id):
            session.update(frame_success)
            if frame_success:
                self.stats.increment("successful_frames")
            else:
                self.stats.increment("failed_frames")

    def cleanup_expired_sessions(self):
        with self._lock:
            expired = [sid for sid, session in self.sessions.items() if session.is_expired(SESSION_TIMEOUT)]
            for sid in expired:
                session = self.sessions.pop(sid)
                session.status = "expired"
                self.stats.increment("expired_sessions")
                self.stats.increment("completed_sessions")

    def get_metrics(self) -> SessionMetrics:
        with self._lock:
            stats = self.stats.get_metrics()
            active = sum(1 for s in self.sessions.values() if s.status == "active")
            if stats["total_sessions"] > 0:
                success_rate = (stats["successful_frames"] / (stats["successful_frames"] + stats["failed_frames"])) * 100 if stats["successful_frames"] + stats["failed_frames"] > 0 else 0
                avg_duration = sum(s.get_duration() for s in self.sessions.values()) / len(self.sessions) if self.sessions else 0
            else:
                success_rate = 0
                avg_duration = 0
            return SessionMetrics(
                total_sessions=stats["total_sessions"],
                active_sessions=active,
                expired_sessions=stats["expired_sessions"],
                failed_sessions=stats["failed_sessions"],
                avg_session_duration=avg_duration,
                success_rate=success_rate
            )

# Initialize FastAPI app with environment-specific settings
app = FastAPI(
    title="Face Authentication API",
    description="Secure face authentication system",
    version="2.5.0",
    debug=DEBUG,
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None
)

# Initialize session manager
session_manager = SessionManager()

# Initialize templates and static files
base_dir = os.getcwd()
templates_dir = os.path.join(base_dir, "templates")
static_dir = os.path.join(base_dir, "static")

os.makedirs(templates_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)

templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

# Initialize face system
face_system = FaceSystem(
    min_detection_confidence=float(os.getenv('MIN_DETECTION_CONFIDENCE', '0.5')),
    target_fps=int(os.getenv('TARGET_FPS', '30')),
    input_size=(
        int(os.getenv('INPUT_WIDTH', '640')),
        int(os.getenv('INPUT_HEIGHT', '480'))
    )
)

# Security middleware with environment-specific settings
if ENVIRONMENT == 'development':
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"]
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"]
    )

# Add security headers in production
if ENVIRONMENT == 'production':
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    if not DEBUG:
        app.add_middleware(HTTPSRedirectMiddleware)

# Request validation models
class OvalGuide(BaseModel):
    x: int = Field(..., description="X coordinate of oval guide")
    y: int = Field(..., description="Y coordinate of oval guide")
    width: int = Field(..., ge=1, description="Width of oval guide")
    height: int = Field(..., ge=1, description="Height of oval guide")
    frame_height: int = Field(..., ge=1, description="Height of the video frame")

    @validator('width', 'height', 'frame_height')
    def validate_dimensions(cls, v):
        if v <= 0:
            raise ValueError("Dimensions must be positive")
        return v

class AuthenticationRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image data")
    oval_guide: OvalGuide = Field(..., description="Oval guide dimensions")
    session_id: Optional[str] = Field(None, description="Session ID for tracking authentication progress")

    @validator('image')
    def validate_image_data(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Image data must be a non-empty string")
        if not v.startswith('data:image') and not v.startswith('/9j/'): # Check both data URL and raw base64
            raise ValueError("Invalid image data format")
        return v

    @validator('session_id')
    def validate_session_id(cls, v):
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError("Session ID must be a non-empty string if provided")
            
            # Allow dev_session format in development mode
            if DEBUG or ENVIRONMENT == 'development':
                if v.startswith('dev_session'):
                    return v
            
            # Production validation
            if not (
                v.startswith('session_') or  # Timestamp-based format
                v.startswith('dev_session') or  # Development format
                len(v.replace('-', '')) == 32  # UUID format (with or without dashes)
            ):
                raise ValueError("Invalid session ID format")
        return v

# JWT token and CSRF models
class Token(BaseModel):
    access_token: str
    token_type: str
    csrf_token: str

# Security functions
def create_csrf_token(session_id: str) -> str:
    """Create a CSRF token with additional entropy."""
    timestamp = str(int(time.time()))
    entropy = secrets.token_hex(16)
    message = f"{session_id}{timestamp}{entropy}".encode()
    return hmac.new(
        CSRF_KEY.encode(),
        message,
        hashlib.sha256
    ).hexdigest()
    
def verify_csrf_token(token: str, session_id: str) -> bool:
    """Verify CSRF token with timing-attack resistance."""
    try:
        if not token or not session_id:
            return False
    
        current_time = int(time.time())
        for timestamp in range(current_time - 1800, current_time + 1):
            entropy = secrets.token_hex(16)
            message = f"{session_id}{timestamp}{entropy}".encode()
            expected_token = hmac.new(
                CSRF_KEY.encode(),
                message,
                hashlib.sha256
            ).hexdigest()
            if hmac.compare_digest(token, expected_token):
                return True
        return False
    except Exception as e:
        logger.error(f"CSRF verification error: {str(e)}")
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token with enhanced security."""
    try:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)

        # Add additional claims
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        # Add entropy to prevent token reuse
        to_encode["jti"] = secrets.token_hex(16)
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        logger.error(f"Token creation error: {str(e)}")
        raise

def create_session_token(session_id: str) -> str:
    """Create a session token with JWT."""
    try:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode = {
            "sub": session_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "session",
            "jti": secrets.token_hex(16)
        }
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    except Exception as e:
        logger.error(f"Session token creation error: {str(e)}")
        raise

def verify_session_token(token: str) -> Optional[str]:
    """Verify session token and return session ID."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "session":
            return None
        return payload.get("sub")
    except jwt.JWTError as e:
        logger.error(f"Session token validation error: {str(e)}")
        return None

# Session verification middleware
async def verify_session(
    request: Request,
    session_token: Optional[str] = Header(None),
    csrf_token: Optional[str] = Header(None)
) -> str:
    """Verify session token and CSRF token."""
    # Skip validation in development mode
    if DEBUG or ENVIRONMENT == 'development':
        logger.debug("Development mode: Skipping session validation")
        return "dev_session"
        
    if not session_token:
        raise HTTPException(
            status_code=401,
            detail="Session token required"
        )
        
    session_id = verify_session_token(session_token)
    if not session_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid session token"
        )
        
    if not csrf_token or not verify_csrf_token(csrf_token, session_id):
        raise HTTPException(
            status_code=403,
            detail="Invalid CSRF token"
        )
        
    return session_id

def process_image(image_data: str) -> np.ndarray:
    """Process base64 image data into OpenCV format"""
    try:
        # Handle empty or invalid input
        if not image_data:
            logger.error("Empty image data received")
            raise ValueError("Empty image data")

        # Clean up the image data string
        if isinstance(image_data, str):
            # Remove any whitespace
            image_data = image_data.strip()
            
            # Extract base64 data
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            elif 'data:image' in image_data:
                image_data = image_data.split(',')[1]
        
        # Convert to image
        try:
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None or frame.size == 0:
                logger.error("Failed to decode image into valid frame")
                raise ValueError("Invalid image data - could not decode frame")
            
            logger.debug(f"Successfully processed frame with shape: {frame.shape}")
            return frame
            
        except (base64.binascii.Error, ValueError) as e:
            logger.error(f"Base64 decoding error: {str(e)}")
            raise ValueError(f"Invalid base64 data: {str(e)}")
            
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid image data: {str(e)}"
        )

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with webcam interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/metrics")
async def get_metrics():
    """Get current system metrics"""
    return session_manager.get_metrics()

@app.get("/sessions")
async def get_sessions():
    """Get all active sessions"""
    return {
        "active_sessions": [
            session.to_dict() 
            for session in session_manager.sessions.values()
            if session.status == "active"
        ]
    }

@app.post("/api/start-session")
async def start_session(request: Request):
    """Start a new authentication session."""
    try:
        client_ip = request.client.host
        
        # Check IP-based rate limiting
        active_sessions = sum(1 for s in session_manager.sessions.values() 
                            if s.client_ip == client_ip and s.status == "active")
        if active_sessions >= MAX_SESSIONS_PER_IP:
            raise HTTPException(
                status_code=429,
                detail="Too many active sessions from this IP"
            )
            
        # Create new session
        session = session_manager.create_session(request)
        session_token = create_session_token(session.id)
        csrf_token = create_csrf_token(session.id)
        
        return JSONResponse({
            "session_id": session.id,
            "session_token": session_token,
            "csrf_token": csrf_token
        })
        
    except Exception as e:
        logger.error(f"Session creation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Could not create session"
        )

@app.post("/api/authenticate")
async def authenticate(
    request: Request,
    auth_request: AuthenticationRequest,
    background_tasks: BackgroundTasks,
    session_id: str = Depends(verify_session)
):
    """Handle face authentication from webcam frame"""
    headers = {
        "Access-Control-Allow-Origin": "*" if DEBUG else ",".join(ALLOWED_ORIGINS),
        "Access-Control-Allow-Methods": "*" if DEBUG else "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Max-Age": "3600",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:;"
    }
    
    session = None
    try:
        client_ip = request.client.host
        logger.info(f"Processing authentication request from {client_ip}")
        
        # Skip session validation in development mode
        if not (DEBUG or ENVIRONMENT == 'development'):
            # Get existing session
            session = session_manager.get_session(session_id, client_ip)
            if not session:
                raise HTTPException(status_code=401, detail="Invalid session")

        # Process frame
        try:
            frame = process_image(auth_request.image)
            logger.debug(f"Successfully processed frame for session {session_id}")
        except Exception as e:
            logger.error(f"Frame processing error for session {session_id}: {str(e)}")
            if session:
                session_manager.update_session(session_id, frame_success=False)
            raise
        
        # Process with face system
        try:
            frame_height = frame.shape[0]
            oval_guide_data = auth_request.oval_guide.model_dump()
            oval_guide_data['frame_height'] = frame_height
            
            processed_frame, results = face_system.process_frame(
                frame,
                draw_debug=False,
                oval_guide=oval_guide_data
            )
            logger.debug(f"Face system results for session {session_id}: {results}")
        except Exception as e:
            logger.error(f"Face processing error for session {session_id}: {str(e)}")
            if session:
                session_manager.update_session(session_id, frame_success=False)
            raise HTTPException(
                status_code=500,
                detail="Error processing face detection"
            )
        
        # Update session with results
        face_detected = results.get('face_detected', False)
        if session:
            session_manager.update_session(session_id, frame_success=face_detected)
        
        # Create new CSRF token for response
        new_csrf_token = create_csrf_token(session_id) if not DEBUG else "dev_csrf_token"
        
        # Prepare response data based on environment
        if DEBUG or ENVIRONMENT == 'development':
            response_data = {
                "success": True,
                "session_id": "dev_session",
                "validation_token": "dev_token",
                "is_live": results.get('is_live', False),
                "face_detected": results.get('face_detected', False),
                "message": results.get('liveness_message', 'Processing...'),
                "confidence": results.get('liveness_confidence', 0.0),
                "recognized_name": results.get('recognized_name'),
                "face_rect": results.get('face_rect'),
                "csrf_token": "dev_csrf_token"
            }
        else:
            response_data = {
                "success": True,
                "session_id": session_id,
                "validation_token": session.validation_token if session else "dev_token",
                "is_live": results.get('is_live', False),
                "face_detected": face_detected,
                "message": results.get('liveness_message', ''),
                "confidence": results.get('liveness_confidence', 0.0),
                "recognized_name": results.get('recognized_name'),
                "face_rect": results.get('face_rect'),
                "csrf_token": new_csrf_token
            }
        
        logger.debug(f"Sending response for session {session_id}: {response_data}")
        return JSONResponse(content=response_data, headers=headers)
        
    except HTTPException as he:
        if session:
            session_manager.update_session(session_id, frame_success=False)
        return JSONResponse(
            status_code=he.status_code,
            content={"detail": he.detail},
            headers=headers
        )
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}", exc_info=True)
        if session:
            session_manager.stats.increment("failed_sessions")
            session_manager.update_session(session_id, frame_success=False)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
            headers=headers
        )

# Error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600"
        }
    )

# Add OPTIONS endpoint handler with explicit CORS headers
@app.options("/api/authenticate")
async def authenticate_options():
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600"
        }
    )

def generate_secure_session_id(client_ip: str) -> Tuple[str, str]:
    """Generate a cryptographically secure session ID with HMAC validation"""
    timestamp = int(time.time() * 1000)
    random_bytes = secrets.token_bytes(16)
    
    # Create base session ID
    base_id = f"{timestamp}.{random_bytes.hex()}"
    
    # Create HMAC using client IP and timestamp for validation
    hmac_data = f"{client_ip}.{timestamp}".encode()
    hmac_signature = hmac.new(
        SECRET_KEY.encode(),
        hmac_data,
        hashlib.sha256
    ).hexdigest()
    
    # Combine into final session ID with validation
    session_id = f"s.{base_id}.{hmac_signature[:8]}"
    
    # Create a validation token
    validation_token = hmac.new(
        SECRET_KEY.encode(),
        session_id.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return session_id, validation_token

def validate_session_id(session_id: str, client_ip: str) -> bool:
    """Validate a session ID's authenticity and timestamp"""
    try:
        # Handle legacy session_* format
        if session_id.startswith('session_'):
            return True  # Legacy sessions are always considered valid
            
        # Handle new secure format (s.timestamp.random.hmac)
        if session_id.startswith('s.'):
            parts = session_id.split('.')
            if len(parts) != 4:
                return False
            
            timestamp_str = parts[1]
            hmac_signature = parts[3]
            
            # Validate timestamp (prevent replay attacks)
            timestamp = int(timestamp_str)
            current_time = int(time.time() * 1000)
            if current_time - timestamp > SESSION_TIMEOUT * 1000:
                return False
            
            # Validate HMAC
            hmac_data = f"{client_ip}.{timestamp}".encode()
            expected_hmac = hmac.new(
                SECRET_KEY.encode(),
                hmac_data,
                hashlib.sha256
            ).hexdigest()[:8]
            
            return hmac.compare_digest(hmac_signature, expected_hmac)
        
        return False
            
    except Exception:
        return False

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv('HOST', '127.0.0.1' if ENVIRONMENT == 'development' else '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))
    
    # Number of worker processes (production only)
    workers = int(os.getenv('WORKERS', '1'))
    
    print(f"Starting server on {host}:{port}")
    if ENVIRONMENT == 'production':
        # Production settings
        uvicorn.run(
            "api:app",
            host=host,
            port=port,
            workers=workers,
            proxy_headers=True,
            forwarded_allow_ips='*',
            access_log=True,
            log_level=log_level.lower(),
            reload=False
        )
    else:
        # Development settings
        uvicorn.run(
            "api:app",
            host=host,
            port=port,
            reload=True,
            access_log=True,
            log_level="debug"
        ) 