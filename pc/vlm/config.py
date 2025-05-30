# Configuration file for the hybrid detection and tracking system (Florence2 + Cutie)

# Network Configuration
IMAGE_PORT = 12345
BBOX_PORT = 12346
JSON_SERVER_HOST = '127.0.0.1'  # localhost
JSON_SERVER_PORT = 65430         # Port for receiving JSON from TargetAgent
ORDER_LISTENER_HOST = '0.0.0.0'  # Listen on all interfaces for external commands
ORDER_PORT = 12347               # Port for receiving switch commands

# Image Server Configuration
IMAGE_SERVER_HOST = '192.168.3.11'  # IP of image_server
IMAGE_SERVER_MESSAGE_PORT = 12348    # Port to send direction commands to image_server
SERVER_HOST_IP = '192.168.3.11'     # Main server IP

# Model Configuration
FLORENCE2_MODEL_PATH = "D:/models"   # Path to Florence2 models
CUTIE_MAX_INTERNAL_SIZE = 480        # Cutie processing resolution
DEFAULT_ONTOLOGY = {"1": "beanbag"}  # Default detection ontology

# System States
STATE_DETECTING = "DETECTING"        # Florence2 detection phase
STATE_TRACKING = "TRACKING"          # Cutie tracking phase  
STATE_SWITCHING = "SWITCHING"        # Target switching phase

# Performance Configuration
TARGET_FPS = 25.0                   # Target display FPS
BBOX_SEND_INTERVAL = 0.1            # Interval for sending bbox data
FPS_CALC_INTERVAL = 2               # Calculate FPS every N seconds
RECEIVED_FPS_INTERVAL = 2           # Calculate received FPS every N seconds
PERFORMANCE_CHECK_INTERVAL = 2.0    # Check performance every N seconds

# Queue Configuration
FRAME_QUEUE_MAXSIZE = 3             # Maximum frames in main queue (keep small for real-time)
PROCESSING_QUEUE_MAXSIZE = 5        # Maximum frames in processing queue
JSON_COMMAND_QUEUE_MAXSIZE = 10     # Maximum JSON commands in queue
SYSTEM_COMMAND_QUEUE_MAXSIZE = 10   # Maximum system commands in queue

# Timeout Configuration
SOCKET_TIMEOUT = 10.0               # Default socket timeout
BBOX_SOCKET_TIMEOUT = 0.5           # Timeout for bbox sending
IMAGE_SOCKET_TIMEOUT = 1.0          # Timeout for image receiving
THREAD_JOIN_TIMEOUT = 3.0           # Timeout for thread joining during shutdown
ASYNC_SWITCH_TIMEOUT = 5.0          # Timeout for async target switch

# Network Optimization (from remote_camera_demo)
NETWORK_CHUNK_SIZE = 16384          # Network receive chunk size
NETWORK_RCVBUF_SIZE = 262144        # Socket receive buffer size
NETWORK_SNDBUF_SIZE = 262144        # Socket send buffer size
ENABLE_TCP_NODELAY = True           # Enable TCP_NODELAY for low latency

# Performance Monitoring
PERFORMANCE_HISTORY_SIZE = 20       # Number of performance samples to keep
FPS_THRESHOLD_RATIO = 0.8           # Use async mode if FPS < target * ratio
GPU_USAGE_THRESHOLD = 90            # GPU usage threshold for async mode
QUEUE_BACKLOG_THRESHOLD = 3         # Queue size threshold for async mode

# GPU Memory Management
ENABLE_GPU_MEMORY_MONITORING = True # Enable GPU memory monitoring
GPU_MEMORY_WARNING_THRESHOLD = 0.9  # Warning when GPU memory > 90%
AUTO_CLEAR_GPU_CACHE = True         # Automatically clear GPU cache on model switch

# Valid Direction Commands
VALID_DIRECTIONS = [
    "forward", "backward", "left", "right",
    "front-left", "front-right", "back-left", "back-right"
]

# Display Configuration
WINDOW_NAME = 'Hybrid Detection & Tracking Viewer'
WAITING_TEXT = "Waiting for JSON data..."
DETECTING_TEXT = "Detecting targets..."
TRACKING_TEXT = "Tracking target..."
SWITCH_PROGRESS_TEXT = "Target switching in progress..."
ASYNC_MODE_TEXT = "(Async Mode)"
SYNC_MODE_TEXT = "(Sync Mode)"

# Mask to Bbox Conversion
BBOX_METHOD = "min_rect"            # "min_rect", "contour", or "center"

# Detection Configuration
DETECTION_RESULT_TIMEOUT = 0.5      # Maximum age of detection results in seconds
TARGET_SWITCH_DEBOUNCE_INTERVAL = 2.0  # Minimum seconds between target switches
MODEL_PATH = "D:/models"             # Path to detection models

# Performance Configuration
BBOX_SEND_INTERVAL = 0.1            # Interval for sending bbox data
FPS_CALC_INTERVAL = 2               # Calculate FPS every N seconds
RECEIVED_FPS_INTERVAL = 2           # Calculate received FPS every N seconds

# Queue Configuration
FRAME_QUEUE_MAXSIZE = 1             # Maximum frames in detection queue
DETECTION_QUEUE_MAXSIZE = 1         # Maximum detection results in queue

# Timeout Configuration
SOCKET_TIMEOUT = 10.0               # Default socket timeout
BBOX_SOCKET_TIMEOUT = 0.5           # Timeout for bbox sending
IMAGE_SOCKET_TIMEOUT = 1.0          # Timeout for image receiving
THREAD_JOIN_TIMEOUT = 3.0           # Timeout for thread joining during shutdown
ASYNC_SWITCH_TIMEOUT = 5.0          # Timeout for async target switch

# Display Configuration
WINDOW_NAME = 'Object Detection Viewer'
WAITING_TEXT = "Waiting for JSON data..."
SWITCH_PROGRESS_TEXT = "Target switching in progress..." 