# Veronica Face Authentication System

A production-ready face authentication system built with Python, featuring real-time face detection, liveness detection, and face recognition capabilities.

## Features

- Real-time face detection and tracking
- Iris-based depth estimation
- Deep learning-based liveness detection
- Anti-spoofing protection
- Face recognition with high accuracy
- Secure session management and token validation
- Rate limiting and DDoS protection
- Cross-platform support (Windows, Linux, macOS)
- Modern web interface with responsive design
- Production-ready configuration
- Comprehensive logging and monitoring
- GPU acceleration support

## Tech Stack

- **Frontend**: JavaScript, WebRTC
- **Backend**: Python, FastAPI
- **ML/CV**: MediaPipe, InsightFace, OpenCV
- **Database**: SQL Server
- **Security**: JWT, CSRF Protection

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- SQL Server
- Required models (downloaded automatically)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Basel-anaya/veronica.git
cd veronica
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp config/.env.example config/.env
# Edit config/.env with your settings
```

5. Initialize the database:
```bash
python scripts/init_db.py
```

6. Start the server:
```bash
python -m veronica.api.routes
```

## Usage

1. Access the web interface at `http://localhost:5000`
2. Position your face within the oval guide
3. Follow the on-screen instructions
4. Wait for authentication result

## Development

### Project Structure
```
veronica/
├── config/
│   ├── .env
│   └── gunicorn_conf.py
├── docs/
│   ├── README.md
│   └── ARCHITECTURE.md
├── src/
│   └── veronica/
│       ├── api/
│       │   └── routes.py
│       └── core/
│           └── face_system.py
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── app.js
├── templates/
│   └── index.html
├── tests/
│   └── ...
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
black src/
flake8 src/
```

## Security

- HTTPS required in production
- Session-based authentication
- CSRF protection
- Rate limiting
- Input validation
- Secure token management

## Performance

- Frame skipping for performance
- GPU acceleration
- Caching mechanisms
- Resource cleanup
- Memory management

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

© Barzan DIG. All Rights Reserved.
Proprietary and Confidential.

## Author

Basel Anaya# Aurora
# Aurora
