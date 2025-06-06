from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
import base64
import cv2
import numpy as np
from main import FaceSystem
import logging
import os
from dotenv import load_dotenv
import secrets
from typing import Optional
import aiofiles
from PIL import Image
import io
import traceback
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI(
    title="Face Authentication API",
    description="Secure face authentication system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly list allowed methods
    allow_headers=["*"],
    expose_headers=["*"]
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Allow all hosts for testing
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Get camera settings from environment variables
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
CAMERA_EXPOSURE = int(os.getenv('CAMERA_EXPOSURE', '-5'))
CAMERA_GAIN = int(os.getenv('CAMERA_GAIN', '0'))
CAMERA_BUFFER_SIZE = int(os.getenv('CAMERA_BUFFER_SIZE', '1'))

# Initialize face system
face_system = FaceSystem(
    min_detection_confidence=float(os.getenv('MIN_DETECTION_CONFIDENCE', '0.5')),
    target_fps=int(os.getenv('TARGET_FPS', '30')),
    input_size=(
        int(os.getenv('INPUT_WIDTH', '640')),
        int(os.getenv('INPUT_HEIGHT', '480'))
    )
)

def verify_request(request: Request):
    """Verify request headers for security."""
    # Only verify content type for now
    content_type = request.headers.get("Content-Type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=400, detail="Content-Type must be application/json")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with webcam interface."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/camera-status")
async def check_camera(request: Request):
    """Check if camera is accessible."""
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            return JSONResponse(content={
                "status": "error",
                "message": f"Failed to open camera {CAMERA_INDEX}"
            })
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return JSONResponse(content={
                "status": "error",
                "message": "Camera opened but failed to read frame"
            })
            
        return JSONResponse(content={
            "status": "success",
            "message": "Camera is accessible"
        })
    except Exception as e:
        logging.error(f"Camera check error: {str(e)}")
        return JSONResponse(content={
            "status": "error",
            "message": f"Error accessing camera: {str(e)}"
        })

@app.options("/authenticate")
async def authenticate_options():
    """Handle CORS preflight requests"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.post("/authenticate")
async def authenticate(request: Request):
    """Handle face authentication from webcam frame."""
    try:
        # Log full request details
        logging.info(f"Request method: {request.method}")
        logging.info(f"Request headers: {dict(request.headers)}")
        
        # Get JSON data from request
        data = await request.json()
        image_data = data.get("image")
        oval_guide = data.get("ovalGuide")
        
        logging.info(f"Has image data: {bool(image_data)}")
        logging.info(f"Oval guide data: {oval_guide}")
        
        if not image_data:
            return JSONResponse(
                status_code=400,
                content={"error": "No image data provided"}
            )
            
        if not oval_guide:
            return JSONResponse(
                status_code=400,
                content={"error": "No oval guide data provided"}
            )
            
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_data)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise HTTPException(status_code=400, detail="Invalid image data")
            
        except Exception as e:
            logging.error(f"Error decoding image: {str(e)}")
            raise HTTPException(status_code=400, detail="Error decoding image")
        
        # Process frame through face system
        processed_frame, results = face_system.process_frame(frame, draw_debug=True, oval_guide=oval_guide)
        
        # Convert numpy types to Python native types
        is_live = bool(results.get("is_live", False))
        confidence = float(results.get("liveness_confidence", 0.0))
        message = str(results.get("liveness_message", "No face detected"))
        recognized_name = str(results.get("recognized_name", "Unknown")) if results.get("recognized_name") else None
        
        # Prepare response with converted types
        response = {
            "success": True,
            "is_live": is_live,
            "confidence": confidence,
            "message": message,
            "recognized_name": recognized_name
        }
        
        # If we have a processed frame, encode it to base64 for display
        if processed_frame is not None:
            _, buffer = cv2.imencode('.jpg', processed_frame)
            processed_image = base64.b64encode(buffer).decode('utf-8')
            response["processed_image"] = f"data:image/jpeg;base64,{processed_image}"
        
        logging.info(f"Authentication response: {response}")
        return JSONResponse(content=response)
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON data"}
        )
    except Exception as e:
        logging.error(f"Authentication error: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/register")
async def register_person(name: str = Form(...), id_card: UploadFile = File(...)):
    """Handle registration of new person with ID card."""
    try:
        logging.info(f"Received registration request for {name}")
        logging.info(f"File info: {id_card.filename}, {id_card.content_type}, size: {id_card.size}")
        
        # Validate file type
        if not id_card.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        # Read image file
        contents = await id_card.read()
        logging.info(f"Read {len(contents)} bytes from uploaded file")
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        logging.info(f"Successfully decoded image, shape: {img.shape}")
        
        try:
            # Add face to database
            success = face_system.register_face_from_id_card(img, name)
            logging.info(f"Registration result: {success}")
            
            if not success:
                raise HTTPException(status_code=400, detail="Could not detect face in ID card")
            
            logging.info(f"Successfully registered {name}")
            return JSONResponse(content={
                "success": True,
                "message": f"Successfully registered {name}"
            })
        except Exception as e:
            logging.error(f"Error in face registration: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=400, detail=f"Face registration failed: {str(e)}")
            
    except HTTPException as he:
        logging.error(f"HTTP error during registration: {str(he)}")
        return JSONResponse(
            status_code=he.status_code,
            content={
                "success": False,
                "message": he.detail
            }
        )
    except Exception as e:
        logging.error(f"Registration error: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": str(e)
            }
        )

if __name__ == "__main__":
    # Create required directories
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    
    # Run the FastAPI application
    if os.getenv('ENVIRONMENT') == 'production':
        uvicorn.run(
            "test:app",
            host="0.0.0.0", 
            port=8002
        )
    else:
        # For development, run on localhost
        uvicorn.run("test:app", host="127.0.0.1", port=8002) 