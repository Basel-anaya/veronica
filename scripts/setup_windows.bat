@echo off
echo Starting Face Authentication System Setup...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.11 or higher.
    echo Download from: https://www.python.org/downloads/
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo pip is not installed! Installing pip...
    python -m ensurepip --default-pip
)

REM Create and activate virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install CUDA Toolkit (optional but recommended)
echo Please ensure you have NVIDIA CUDA Toolkit installed for GPU support
echo Download from: https://developer.nvidia.com/cuda-downloads
timeout /t 5

REM Install required packages
echo Installing required packages...
pip install -r requirements.txt

:: Install Visual C++ Redistributable
echo Installing Visual C++ Redistributable...
curl -O https://aka.ms/vs/17/release/vc_redist.x64.exe
vc_redist.x64.exe /quiet /norestart

:: Install Visual C++ Build Tools (required for dlib)
echo Please ensure you have Visual Studio Build Tools installed
echo If not installed, download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo Select "Desktop development with C++" workload during installation
pause

:: Install core dependencies
echo Installing core dependencies...
pip install numpy==1.26.2
pip install opencv-python-headless
pip install dlib
pip install mediapipe
pip install insightface
pip install fastapi
pip install uvicorn
pip install python-dotenv
pip install pyodbc
pip install scipy
pip install Pillow
pip install aiofiles
pip install jinja2

:: Install CUDA support
echo Installing CUDA support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

:: Install web dependencies
echo Installing web dependencies...
pip install python-multipart==0.0.6
pip install starlette

:: Install development dependencies
echo Installing development dependencies...
pip install python-jose==3.3.0
pip install passlib

:: Install production dependencies
echo Installing production dependencies...
pip install gunicorn==21.2.0
pip install waitress==2.1.2
pip install pydantic==2.5.2
pip install onnxruntime-gpu==1.16.3

:: Create required directories
echo Creating required directories...
if not exist models\buffalo_l mkdir models\buffalo_l
if not exist faces mkdir faces
if not exist static\css mkdir static\css
if not exist static\js mkdir static\js
if not exist templates mkdir templates

:: Download required models
echo Downloading required models...
powershell -Command "Invoke-WebRequest -Uri https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -OutFile models\buffalo_l.zip"
powershell -Command "Expand-Archive -Path models\buffalo_l.zip -DestinationPath models\buffalo_l -Force"
del models\buffalo_l.zip

powershell -Command "Invoke-WebRequest -Uri http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -OutFile shape_predictor_68_face_landmarks.dat.bz2"
:: Note: Windows users will need 7-Zip or similar to extract .bz2 files
echo Please extract shape_predictor_68_face_landmarks.dat.bz2 manually using 7-Zip or similar tool

:: Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file with default settings...
    (
        echo ENVIRONMENT=development
        echo DEBUG=True
        echo LOG_LEVEL=INFO
        echo HOST=0.0.0.0
        echo PORT=5000
        echo ALLOWED_ORIGINS=http://localhost:5000,http://127.0.0.1:5000
        echo ALLOWED_HOSTS=*
        echo SESSION_TIMEOUT=30
        echo MIN_FRAME_INTERVAL=0.2
        echo MAX_SESSIONS_PER_IP=3
        echo RATE_LIMIT_WINDOW=60
        echo MAX_REQUESTS_PER_WINDOW=100
        echo MIN_DETECTION_CONFIDENCE=0.5
        echo TARGET_FPS=30
        echo INPUT_WIDTH=640
        echo INPUT_HEIGHT=480
    ) > .env
)

REM Check if OpenCV is properly installed
python -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo WARNING: OpenCV installation might have issues. Please ensure your system has the required Visual C++ redistributables.
    echo Download Visual C++ redistributable from: https://aka.ms/vs/17/release/vc_redist.x64.exe
)

REM Check if MediaPipe is properly installed
python -c "import mediapipe" >nul 2>&1
if errorlevel 1 (
    echo WARNING: MediaPipe installation might have issues. Please ensure you have the correct Python version (3.8-3.11).
)

echo.
echo Setup completed successfully!
echo.
echo To start the server, run:
echo     python -m uvicorn api:app --host 0.0.0.0 --port 5000 --reload
echo.
echo For production deployment, set ENVIRONMENT=production in .env file
echo and use:
echo     python -m uvicorn api:app --host 0.0.0.0 --port 5000 --workers 4
echo.

REM Keep the window open
pause 