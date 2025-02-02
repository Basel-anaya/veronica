#! /bin/bash

# Set environment variables
export ENVIRONMENT="production"
export WORKERS_PER_CORE=1
export WEB_CONCURRENCY=2
export HOST=0.0.0.0
export PORT=5000
export LOG_LEVEL=info

# Start Gunicorn with the FastAPI app
exec gunicorn -k uvicorn.workers.UvicornWorker -c gunicorn_conf.py api:app 