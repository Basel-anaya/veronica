import multiprocessing
import os

# Set OpenBLAS thread count
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes - optimized for modern systems
workers = multiprocessing.cpu_count() * 2  # Dynamic worker count based on CPU cores
worker_class = 'uvicorn.workers.UvicornWorker'
worker_connections = 2000  # Increased for better concurrency
timeout = 60  # Increased timeout
keepalive = 5  # Increased keepalive

# Preload application
preload_app = True

# Process naming
proc_name = 'veronica'
pythonpath = '.'

# Logging
accesslog = 'logs/access.log'
errorlog = 'logs/error.log'
loglevel = 'info'

# Resource management - optimized for modern systems
max_requests = 2000  # Increased max requests
max_requests_jitter = 100
graceful_timeout = 60
limit_request_line = 8192
limit_request_fields = 200
limit_request_field_size = 16384

# Development settings
reload = False
reload_engine = 'auto'

# SSL (if needed)
# keyfile = 'ssl/key.pem'
# certfile = 'ssl/cert.pem'

def on_starting(server):
    """
    Hook to set process title and environment variables
    """
    server.log.info("Starting Veronica Face Authentication Server")
    
    # Set environment variables for better resource management
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU mode
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Limit OpenBLAS threads
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Set depth detection thresholds
    os.environ['DEPTH_MIN_THRESHOLD'] = '300'  # 30cm
    os.environ['DEPTH_MAX_THRESHOLD'] = '500'  # 80cm
    os.environ['DEPTH_TOLERANCE'] = '100'      # 10cm tolerance

def post_fork(server, worker):
    """
    Hook to set worker process title and limit memory
    """
    server.log.info(f"Worker spawned (pid: {worker.pid})")

    # Set resource limits more conservatively but suitable for modern systems
    try:
        import resource
        # Set max number of processes
        resource.setrlimit(resource.RLIMIT_NPROC, (1024, 1024))  # Increased process limit
        # Set max number of open files
        resource.setrlimit(resource.RLIMIT_NOFILE, (8192, 8192))  # Increased file limit
        # Set max address space (bytes)
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024 * 16, -1))  # 16GB
    except Exception as e:
        server.log.warning(f"Failed to set resource limits: {e}")

def worker_abort(worker):
    """
    Hook for worker failure
    """
    worker.log.warning(f"Worker {worker.pid} aborted") 