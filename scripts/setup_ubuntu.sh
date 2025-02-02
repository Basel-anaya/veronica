#!/bin/bash

echo "Setting up Face Recognition System for Ubuntu..."

# Check for Python installation
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed!"
    echo "Installing Python3..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv
fi

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install system dependencies
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    software-properties-common \
    curl \
    gnupg \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0

# Install CUDA (optional but recommended)
echo "Installing CUDA dependencies..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers not found. Installing CUDA Toolkit..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update
    sudo apt install -y cuda-toolkit-12-3
    rm cuda-keyring_1.1-1_all.deb
fi

# Install SQL Server ODBC Driver
echo "Installing SQL Server ODBC Driver..."
if [ ! -f /etc/apt/sources.list.d/mssql-release.list ]; then
    curl https://packages.microsoft.com/keys/microsoft.asc | sudo tee /etc/apt/trusted.gpg.d/microsoft.asc
    curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list
    sudo apt update
    sudo ACCEPT_EULA=Y apt install -y msodbcsql18
    sudo ACCEPT_EULA=Y apt install -y mssql-tools18
    sudo apt install -y unixodbc-dev
fi

# Install core dependencies
echo "Installing core dependencies..."
pip install numpy==1.26.2
pip install opencv-python-headless==4.8.1.78
pip install dlib
pip install mediapipe==0.10.8
pip install insightface
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install python-dotenv==1.0.0
pip install pyodbc
pip install scipy
pip install Pillow
pip install aiofiles
pip install jinja2

# Install CUDA support
echo "Installing CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install web dependencies
echo "Installing web dependencies..."
pip install python-multipart==0.0.6
pip install starlette

# Install development dependencies
echo "Installing development dependencies..."
pip install python-jose==3.3.0
pip install passlib

# Install production dependencies
echo "Installing production dependencies..."
pip install gunicorn==21.2.0
pip install pydantic==2.5.2
pip install onnxruntime-gpu==1.16.3

# Create required directories
echo "Creating required directories..."
mkdir -p models/buffalo_l
mkdir -p faces
mkdir -p static/css
mkdir -p static/js
mkdir -p templates

# Download required models
echo "Downloading required models..."
wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -O models/buffalo_l.zip
unzip models/buffalo_l.zip -d models/buffalo_l/
rm models/buffalo_l.zip

wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOL
# Database Configuration
conn_str='DRIVER={ODBC Driver 18 for SQL Server};SERVER=your_server;DATABASE=your_db;UID=your_user;PWD=your_password;TrustServerCertificate=yes;Encrypt=no;'

# Server Configuration
PORT=8000
WORKERS=2
ENVIRONMENT=production

# CORS Settings
ALLOWED_ORIGINS=*
ALLOWED_HOSTS=*
EOL
fi

# Make start script executable
echo "Creating start script..."
cat > start.sh << EOL
#!/bin/bash
source venv/bin/activate
exec gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 2 api:app
EOL
chmod +x start.sh

echo "Setup completed successfully!"
echo "To start the server in development mode: python -m uvicorn api:app --host 0.0.0.0 --port 8000"
echo "To start the server in production mode: ./start.sh" 