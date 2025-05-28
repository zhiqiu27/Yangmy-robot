import cv2
import numpy as np
import socket
import struct # To send/receive data size
import time
import threading
import queue
import json
from object_detect import initialize_detection_model, run_detection, draw_detections_on_frame, create_ontology_from_caption, update_model_ontology
import torch

# Global variables
img_display_global = None # Stores the latest raw frame from server for immediate display
latest_detections_for_bbox_send = None # Stores sv.Detections object from detection thread or queue
# --- NEW: Add frame timestamp tracking ---
latest_frame_timestamp = 0  # Timestamp of the latest received frame
latest_detection_timestamp = 0  # Timestamp of the latest detection result
DETECTION_RESULT_TIMEOUT = 0.5  # Maximum age of detection results in seconds
# --- END NEW ---

# --- MODIFIED/ADDED Globals for target switching ---
ONTOLOGY_FOR_VIEWER = None # Will be updated on target switch
g_model_object = None      # To store the model object globally - LOADED ONCE
g_device_object = None     # To store the device object globally - LOADED ONCE

g_target_entities_list = []
g_current_target_index = 0
g_target_switch_event = threading.Event()
g_order_listener_lock = threading.Lock()
current_detection_thread_handler = None # To store the handle of the detection thread

# --- NEW: Globals for pre-startup and activation ---
g_detection_active = False  # Flag to control whether detection should run
g_detection_activation_lock = threading.Lock()
g_system_ready_event = threading.Event()   # Event to signal system is ready for JSON

# --- NEW: Target switch debouncing ---
g_last_target_switch_time = 0  # Timestamp of last target switch
TARGET_SWITCH_DEBOUNCE_INTERVAL = 2.0  # Minimum seconds between target switches

# --- NEW: Async target switching ---
g_target_switch_thread = None  # Handle for async target switch thread
g_target_switch_in_progress = False  # Flag to indicate if target switch is in progress
g_target_switch_lock = threading.Lock()  # Lock for target switch state

IMAGE_PORT = 12345
BBOX_PORT = 12346
JSON_SERVER_HOST = '127.0.0.1'  # localhost
JSON_SERVER_PORT = 65430         # Port for receiving JSON from TargetAgent
ORDER_LISTENER_HOST = '0.0.0.0'  # Listen on all interfaces for external commands
ORDER_PORT = 12347 # New port for receiving switch commands

# --- NEW: Configuration for sending messages to image_server ---
IMAGE_SERVER_HOST = '192.168.3.11'  # IP of image_server
IMAGE_SERVER_MESSAGE_PORT = 12348    # Port to send direction commands to image_server
# --- END NEW ---

# Queues for communication between main thread and detection thread
FRAME_FOR_DETECTION_QUEUE = queue.Queue(maxsize=1)  # Frames to be processed
DETECTION_RESULTS_QUEUE = queue.Queue(maxsize=1)    # Detection results

DETECTION_THREAD_STOP_EVENT = threading.Event()

def receive_all(sock, n):
    data = bytearray()
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
            if not packet:
                print("Connection closed by server while receiving.")
                return None
            data.extend(packet)
        except socket.timeout:
            print(f"Socket timeout during receive_all for {n} bytes. Server may be unresponsive.")
            return None
        except socket.error as e:
            print(f"Socket error during receive_all: {e}")
            return None
    return data

def detection_worker(model, device, ontology_for_detection):
    """Thread that performs object detection on frames from a queue."""
    print(f"Detection thread started for ontology: {ontology_for_detection}")
    detection_count = 0
    last_fps_calc_time = time.time()
    fps_calc_interval = 2 # seconds (calculate FPS every 2 seconds, for example)

    try:
        while not DETECTION_THREAD_STOP_EVENT.is_set():
            try:
                # --- NEW: Check if detection is active ---
                with g_detection_activation_lock:
                    detection_active = g_detection_active
                
                if not detection_active:
                    # Wait for activation or stop event, but check stop event frequently
                    for _ in range(10):  # Check 10 times over 1 second
                        if DETECTION_THREAD_STOP_EVENT.is_set():
                            break
                        time.sleep(0.1)
                    continue
                # --- END NEW ---
                
                # --- NEW: Get current model and device from globals ---
                current_model = g_model_object
                current_device = g_device_object
                
                if current_model is None or current_device is None:
                    # Model not ready yet, wait a bit but check stop event
                    for _ in range(10):  # Check 10 times over 1 second
                        if DETECTION_THREAD_STOP_EVENT.is_set():
                            break
                        time.sleep(0.1)
                    continue
                # --- END NEW ---
                
                try:
                    raw_frame_to_detect = FRAME_FOR_DETECTION_QUEUE.get(timeout=0.1)
                except queue.Empty:
                    continue

                if raw_frame_to_detect is None: 
                    print("Detection thread: Received sentinel, exiting...")
                    try:
                        DETECTION_RESULTS_QUEUE.put(None) 
                    except queue.Full:
                        pass
                    break
                
                # Check stop event before expensive detection operation
                if DETECTION_THREAD_STOP_EVENT.is_set():
                    print("Detection thread: Stop event detected before detection, exiting...")
                    break
                
                try:
                    detections_obj = run_detection(raw_frame_to_detect, current_model)  # Use current_model instead of model
                except Exception as e:
                    print(f"Detection worker: Error during detection: {e}")
                    detections_obj = None
                
                # Check stop event after detection (in case it was set during detection)
                if DETECTION_THREAD_STOP_EVENT.is_set():
                    print("Detection thread: Stop event detected after detection, exiting...")
                    break
                    
                detection_count += 1 # Increment even if detections_obj is None (means an attempt was made)

                if detections_obj is not None:
                    try:
                        DETECTION_RESULTS_QUEUE.put_nowait(detections_obj) 
                    except queue.Full:
                        pass 

                try:
                    FRAME_FOR_DETECTION_QUEUE.task_done()
                except ValueError:
                    # task_done() called more times than there were items placed in the queue
                    pass

                # Calculate and print detection FPS periodically
                current_time = time.time()
                if current_time - last_fps_calc_time >= fps_calc_interval:
                    fps = detection_count / (current_time - last_fps_calc_time)
                    print(f"[Detection Thread] FPS: {fps:.2f}")
                    detection_count = 0
                    last_fps_calc_time = current_time
                    
            except Exception as e:
                print(f"Detection worker: Error in main loop iteration: {e}")
                # Continue the loop instead of breaking, to maintain robustness
                time.sleep(0.1)
                continue

    except Exception as e:
        print(f"Detection worker: Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Detection thread finished.")
        # Clean up any remaining queue items
        try:
            while not FRAME_FOR_DETECTION_QUEUE.empty():
                try:
                    FRAME_FOR_DETECTION_QUEUE.get_nowait()
                    FRAME_FOR_DETECTION_QUEUE.task_done()
                except (queue.Empty, ValueError):
                    break
        except Exception as e:
            print(f"Detection thread cleanup error: {e}")

def order_listener_thread_func():
    """Listens for commands on ORDER_PORT to switch targets."""
    global g_current_target_index, g_target_entities_list
    print(f"Order listener starting on {ORDER_LISTENER_HOST}:{ORDER_PORT}")
    order_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    order_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        order_socket.bind((ORDER_LISTENER_HOST, ORDER_PORT))
        order_socket.listen(1)
        print(f"Order listener: Listening on {ORDER_LISTENER_HOST}:{ORDER_PORT}")

        while True: # Keep listening for new connections
            try:
                conn, addr = order_socket.accept()
                with conn:
                    print(f"Order listener: Accepted connection from {addr}")
                    command_bytes = conn.recv(1024) # Receive command
                    if not command_bytes:
                        continue
                    command = command_bytes.decode('utf-8').strip()
                    print(f"Order listener: Received command '{command}'")

                    if command == "NEXT_TARGET":
                        with g_order_listener_lock:
                            if not g_target_entities_list or len(g_target_entities_list) < 2:
                                print("Order listener: Not enough targets to switch.")
                                conn.sendall(b"ERROR: Not enough targets\n")
                            elif g_current_target_index == 1: # Already on the second target
                                print("Order listener: Already on the second target or only one target to switch to.")
                                conn.sendall(b"INFO: Already on second target or no further switch\n")
                            else: # Switch to index 1 if currently on 0
                                print("Order listener: Switching to next target (index 1).")
                                g_current_target_index = 1
                                g_target_switch_event.set() # Signal main thread
                                conn.sendall(b"OK: Switching to next target\n")
                        print(f"Order listener: Target index set to {g_current_target_index}, event set.")
                    else:
                        print(f"Order listener: Unknown command '{command}'")
                        conn.sendall(f"ERROR: Unknown command {command}\n".encode('utf-8'))
            except socket.timeout:
                continue # Allow checking a potential stop condition for the listener if added later
            except socket.error as e:
                print(f"Order listener: Socket error: {e}. May stop listening if critical.")
                # Consider whether to break the loop on certain errors
                time.sleep(1) # Avoid busy-looping on repeated errors

    except Exception as e:
        print(f"Order listener: Critical error {e}. Thread stopping.")
    finally:
        print("Order listener: Closing socket.")
        order_socket.close()

def json_receiver_thread_func():
    """Background thread that waits for JSON data from TargetAgent."""
    global g_target_entities_list, g_current_target_index, ONTOLOGY_FOR_VIEWER
    global g_model_object, g_device_object, g_detection_active
    
    print(f"JSON receiver thread starting. Waiting for system ready signal...")
    
    # Wait for system to be ready before starting JSON server
    g_system_ready_event.wait()
    print(f"System ready. Starting JSON server on {JSON_SERVER_HOST}:{JSON_SERVER_PORT}")
    
    json_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    json_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        json_server_socket.bind((JSON_SERVER_HOST, JSON_SERVER_PORT))
        json_server_socket.listen(1)
        print(f"JSON server listening on {JSON_SERVER_HOST}:{JSON_SERVER_PORT}. Waiting for TargetAgent connection...")

        while True:  # Allow multiple JSON updates
            try:
                conn, addr = json_server_socket.accept()
                with conn:
                    print(f"JSON server: Accepted connection from {addr}")
                    
                    # Receive the length of the JSON data first
                    raw_msglen = receive_all(conn, 4) 
                    if not raw_msglen:
                        print("JSON server: Failed to receive message length.")
                        continue
                    
                    msglen = struct.unpack('>I', raw_msglen)[0]
                    print(f"JSON server: Receiving {msglen} bytes of JSON data.")
                    
                    target_agent_json_output = None
                    if msglen == 0:
                        print("JSON server: Received zero length for JSON data. Assuming no target.")
                        target_agent_json_output = "{}"
                    elif msglen > 0:
                        json_data_bytes = receive_all(conn, msglen)
                        if not json_data_bytes:
                            print("JSON server: Failed to receive JSON data.")
                            continue
                        else:
                            target_agent_json_output = json_data_bytes.decode('utf-8')
                            print(f"JSON server: Received data: {target_agent_json_output}")
                    else:
                        print(f"JSON server: Received invalid message length {msglen}.")
                        continue
                    
                    # Process the JSON data
                    if target_agent_json_output:
                        success = process_json_data(target_agent_json_output)
                        if success:
                            print("JSON data processed successfully. Detection activated.")
                        else:
                            print("Failed to process JSON data.")
                            
            except socket.error as e:
                print(f"JSON server: Socket error: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"JSON server: Unexpected error: {e}")
                time.sleep(1)
                
    except Exception as e:
        print(f"JSON server: Critical error {e}. Thread stopping.")
    finally:
        print("JSON server: Closing socket.")
        json_server_socket.close()

def send_direction_to_image_server(direction):
    """Send a direction command to image_server on port 12348"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5.0)
            s.connect((IMAGE_SERVER_HOST, IMAGE_SERVER_MESSAGE_PORT))
            s.sendall(direction.encode('utf-8'))
            print(f"[Direction Sender] Sent direction to image_server: '{direction}'")
            
            # Receive acknowledgment
            response = s.recv(1024)
            print(f"[Direction Sender] Received response: {response.decode('utf-8')}")
            return True
    except Exception as e:
        print(f"[Direction Sender] Failed to send direction '{direction}': {e}")
        return False

def handle_direction_command(direction_value):
    """Handle direction commands from DirectionAgent."""
    print(f"[Direction Handler] Processing direction command: '{direction_value}'")
    
    # Valid directions from DirectionAgent
    valid_directions = [
        "forward", "backward", "left", "right",
        "front-left", "front-right", "back-left", "back-right"
    ]
    
    if direction_value not in valid_directions:
        print(f"[Direction Handler] Invalid direction: '{direction_value}'. Valid directions: {valid_directions}")
        return False
    
    print(f"[Direction Handler] Robot should move: {direction_value}")
    
    # Send direction to image_server
    send_direction_to_image_server(direction_value)
    
    # TODO: Add robot control interface here
    # Examples:
    # - Send command to robot controller
    # - Update robot state
    # - Send movement commands via another socket/API
    
    return True

def process_json_data(target_agent_json_output):
    """Process received JSON data and activate detection if successful."""
    global g_target_entities_list, g_current_target_index, ONTOLOGY_FOR_VIEWER
    global g_model_object, g_device_object, g_detection_active, current_detection_thread_handler
    
    try:
        data = json.loads(target_agent_json_output)
        
        # Check if this is a DirectionAgent JSON (contains "direction" field)
        if "direction" in data:
            direction_value = data.get("direction", "")
            print(f"[JSON Processor] Received DirectionAgent JSON with direction: '{direction_value}'")
            
            # Handle direction command
            handle_direction_command(direction_value)
            return True  # Successfully processed direction command
        
        # Check if this is a TargetAgent JSON (contains "target_entities" field)
        elif "target_entities" in data:
            print(f"[JSON Processor] Received TargetAgent JSON")
            new_target_entities_list = data.get("target_entities", [])

            if not new_target_entities_list or not isinstance(new_target_entities_list, list) or len(new_target_entities_list) == 0:
                print(f"No valid target entities found in JSON: {target_agent_json_output}")
                return False
            
            # Update target list
            with g_order_listener_lock:
                g_target_entities_list = new_target_entities_list
                g_current_target_index = 0
            
            # Use the first entity as the primary target for detection
            target_object_name = g_target_entities_list[g_current_target_index]
            target_object_caption = f"a {target_object_name}"
            initial_ontology_caption = {target_object_name: target_object_caption}
            
            print(f"Using target object from TargetAgent: '{target_object_name}' with caption: '{target_object_caption}'")
            if len(g_target_entities_list) > 1:
                print(f"Secondary entity from TargetAgent: '{g_target_entities_list[1]}'")

            # Initialize or re-initialize detection model if needed
            if not g_model_object or not g_device_object:
                print("Performing ONE-TIME initialization of detection model and device...")
                g_model_object, g_device_object, initial_ontology_obj = initialize_detection_model(
                    model_path="D:/models", 
                    ontology_caption=initial_ontology_caption
                )
                if not g_model_object or not g_device_object:
                    print("Failed to initialize model/device.")
                    return False
                ONTOLOGY_FOR_VIEWER = initial_ontology_obj
                print("Detection model and device initialized.")
            else:
                # Model already exists, reload it with new ontology
                print("Model already exists. Reloading model with new ontology...")
                success, updated_model, updated_device, updated_ontology = update_model_ontology(initial_ontology_caption)
                if success and updated_model and updated_device and updated_ontology:
                    g_model_object = updated_model
                    g_device_object = updated_device
                    ONTOLOGY_FOR_VIEWER = updated_ontology
                    print("Model reloaded with new ontology.")
                else:
                    print("Failed to reload model. Keeping existing model.")
                    return False
            
            # Activate detection
            with g_detection_activation_lock:
                g_detection_active = True
            
            print("Detection activated successfully.")
            return True
        
        else:
            print(f"[JSON Processor] Unknown JSON format received: {target_agent_json_output}")
            print("Expected either 'target_entities' (TargetAgent) or 'direction' (DirectionAgent) field")
            return False
        
    except json.JSONDecodeError:
        print(f"Invalid JSON input: {target_agent_json_output}")
        return False
    except Exception as e:
        print(f"Error processing JSON output '{target_agent_json_output}': {e}")
        return False

def main():
    global img_display_global, ONTOLOGY_FOR_VIEWER, latest_detections_for_bbox_send
    global g_target_entities_list, g_current_target_index, current_detection_thread_handler
    global g_model_object, g_device_object, g_detection_active, g_last_target_switch_time
    global g_target_switch_thread, g_target_switch_in_progress

    print("=== Starting Object Detection Viewer with Pre-startup Threads ===")
    
    # --- STEP 1: Initialize basic components (no detection model yet) ---
    print("Step 1: Initializing basic components...")
    
    # Clear all events
    g_system_ready_event.clear()
    g_target_switch_event.clear()
    DETECTION_THREAD_STOP_EVENT.clear()
    
    # Initialize detection state
    with g_detection_activation_lock:
        g_detection_active = False
    
    # --- STEP 2: Start background threads ---
    print("Step 2: Starting background threads...")
    
    # Start JSON receiver thread (will wait for system ready signal)
    json_thread = threading.Thread(target=json_receiver_thread_func, name="JSONReceiverThread")
    json_thread.daemon = True
    json_thread.start()
    print("JSON receiver thread started (waiting for system ready).")
    
    # Start order listener thread
    order_thread = threading.Thread(target=order_listener_thread_func, name="OrderListenerThread")
    order_thread.daemon = True
    order_thread.start()
    print("Order listener thread started.")
    
    # --- STEP 3: Start detection thread (but inactive) ---
    print("Step 3: Starting detection thread (inactive mode)...")
    
    # Start detection thread with dummy parameters (will be updated when JSON arrives)
    # The thread will wait for activation
    current_detection_thread_handler = threading.Thread(
        target=detection_worker, 
        args=(None, None, None),  # Dummy parameters, will be updated
        name="DetectionWorkerThread"
    )
    current_detection_thread_handler.daemon = True
    current_detection_thread_handler.start()
    print("Detection thread started in inactive mode.")
    
    # --- STEP 4: Connect to image and bbox servers ---
    print("Step 4: Connecting to image and bbox servers...")
    
    server_host_ip = '192.168.3.11'
    image_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bbox_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    image_socket.settimeout(10.0)
    bbox_socket.settimeout(10.0)

    try:
        print(f"Connecting to IMAGE server at {server_host_ip}:{IMAGE_PORT}...")
        image_socket.connect((server_host_ip, IMAGE_PORT))
        print("Connected to IMAGE server.")
        print(f"Connecting to BBOX server at {server_host_ip}:{BBOX_PORT}...")
        bbox_socket.connect((server_host_ip, BBOX_PORT))
        print("Connected to BBOX server.")
    except Exception as e:
        print(f"Error connecting to servers: {e}")
        print("Shutting down due to connection failure...")
        shutdown_all_threads()
        return

    # --- STEP 5: Signal system ready and start main loop ---
    print("Step 5: System ready. Signaling JSON receiver to start listening...")
    g_system_ready_event.set()  # Signal JSON receiver to start
    
    cv2.namedWindow('Object Detection Viewer')
    print("=== System started successfully ===")
    print("Waiting for JSON data to activate detection...")
    print("You can also send images - they will be displayed but not processed until JSON is received.")

    last_bbox_send_time = time.time()
    bbox_send_interval = 0.1
    received_frame_count = 0
    last_received_fps_time = time.time()
    received_fps_interval = 2

    try:
        while True:
            # --- Handle Target Switching (ASYNC - Non-blocking) ---
            if g_target_switch_event.is_set():
                print("Main loop: Detected target switch event.")
                try:
                    # Check debounce interval
                    current_time = time.time()
                    if current_time - g_last_target_switch_time < TARGET_SWITCH_DEBOUNCE_INTERVAL:
                        print(f"Target switch ignored - debounce interval not met ({TARGET_SWITCH_DEBOUNCE_INTERVAL}s)")
                        g_target_switch_event.clear()
                        continue
                        
                    with g_detection_activation_lock:
                        if g_detection_active:
                            g_last_target_switch_time = current_time  # Update timestamp before switch
                            print("Starting ASYNC target switch - video will continue...")
                            
                            # Start async target switch (non-blocking)
                            switch_started = start_async_target_switch()
                            if switch_started:
                                print("Async target switch initiated successfully")
                            else:
                                print("Failed to start async target switch (may be in progress)")
                        else:
                            print("Target switch ignored - detection not active yet.")
                except Exception as e:
                    print(f"ERROR: Failed to initiate async target switch: {e}")
                    print("Main loop will continue running...")
                    import traceback
                    traceback.print_exc()
                finally:
                    g_target_switch_event.clear()  # Always clear the event
            
            # --- Receive and display images ---
            image_socket.settimeout(1.0)
            img_size_data = receive_all(image_socket, 4)
            if not img_size_data:
                print("IMAGE server connection issue (img size). Closing.")
                break
            img_size = struct.unpack('>I', img_size_data)[0]

            if img_size == 0:
                time.sleep(0.01)
            else:
                img_data_jpeg = receive_all(image_socket, img_size)
                if not img_data_jpeg:
                    print("IMAGE server connection issue (img data). Closing.")
                    break
                try:
                    img_np_arr = np.frombuffer(img_data_jpeg, dtype=np.uint8)
                    decoded_frame = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
                    if decoded_frame is not None:
                        img_display_global = decoded_frame
                        received_frame_count += 1
                        
                        # Only send to detection queue if detection is active
                        with g_detection_activation_lock:
                            if g_detection_active:
                                try:
                                    FRAME_FOR_DETECTION_QUEUE.put_nowait(img_display_global.copy())
                                except queue.Full:
                                    pass  # Skip frame if queue is full
                    else:
                        print("Failed to decode image.")
                except Exception as e:
                    print(f"Error decoding image: {e}")

            # --- Calculate and Display Received Frames FPS ---
            current_time_fps_calc = time.time()
            if current_time_fps_calc - last_received_fps_time >= received_fps_interval:
                fps = received_frame_count / (current_time_fps_calc - last_received_fps_time)
                print(f"[Main Thread - Image Reception] FPS: {fps:.2f}")
                received_frame_count = 0
                last_received_fps_time = current_time_fps_calc

            # --- Display Logic ---
            display_frame_final = None
            if img_display_global is not None:
                display_frame_final = img_display_global.copy()

                # Check for detection results only if detection is active
                with g_detection_activation_lock:
                    if g_detection_active:
                        try:
                            new_detections_from_queue = DETECTION_RESULTS_QUEUE.get_nowait()
                            if new_detections_from_queue is not None:
                                latest_detections_for_bbox_send = new_detections_from_queue
                        except queue.Empty:
                            pass

                        # Draw detections if available
                        if latest_detections_for_bbox_send and ONTOLOGY_FOR_VIEWER is not None:
                            draw_detections_on_frame(display_frame_final, latest_detections_for_bbox_send, ONTOLOGY_FOR_VIEWER)
                    else:
                        # Add text overlay indicating waiting for JSON
                        cv2.putText(display_frame_final, "Waiting for JSON data...", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # --- NEW: Add target switch status overlay ---
                with g_target_switch_lock:
                    if g_target_switch_in_progress:
                        cv2.putText(display_frame_final, "Target switching in progress...", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                # --- END NEW ---
            
            if display_frame_final is not None:
                cv2.imshow('Object Detection Viewer', display_frame_final)
                cv2.waitKey(1)
            # --- Bbox Sending Logic (only if detection is active) ---
            current_time = time.time()
            if current_time - last_bbox_send_time > bbox_send_interval:
                with g_detection_activation_lock:
                    if g_detection_active and latest_detections_for_bbox_send and \
                       hasattr(latest_detections_for_bbox_send, 'xyxy') and latest_detections_for_bbox_send.xyxy is not None and \
                       len(latest_detections_for_bbox_send.xyxy) > 0:
                        first_box = latest_detections_for_bbox_send.xyxy[0]
                        payload_str = f"{int(first_box[0])},{int(first_box[1])},{int(first_box[2])},{int(first_box[3])}"
                        payload_bytes = payload_str.encode('utf-8')
                        try:
                            bbox_socket.settimeout(0.5)
                            bbox_socket.sendall(struct.pack('>I', len(payload_bytes)))
                            bbox_socket.sendall(payload_bytes)
                            last_bbox_send_time = current_time
                        except socket.timeout:
                            print("Timeout sending bbox on BBOX_SOCKET.")
                        except socket.error as e:
                            print(f"Error sending bbox on BBOX_SOCKET: {e}")
                    else:
                        # Send zero-size signal
                        try:
                            bbox_socket.settimeout(0.5)
                            bbox_socket.sendall(struct.pack('>I', 0))
                            last_bbox_send_time = current_time
                        except socket.timeout:
                            print("Timeout sending zero-size bbox signal.")
                        except socket.error as e:
                            print(f"Error sending zero-size bbox: {e}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
    
    except Exception as e:
        print(f"Main loop error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Shutting down...")
        shutdown_all_threads()
        if image_socket: image_socket.close()
        if bbox_socket: bbox_socket.close()
        cv2.destroyAllWindows()
        print("Client shut down complete.")

def async_target_switch():
    """Asynchronous target switching that runs in a separate thread."""
    global g_target_switch_in_progress, g_target_switch_thread
    
    try:
        with g_target_switch_lock:
            g_target_switch_in_progress = True
        
        print("[ASYNC] Target switch thread started - video will continue uninterrupted")
        
        # Call the original target switch function
        handle_target_switch()
        
        print("[ASYNC] Target switch thread completed successfully")
        
    except Exception as e:
        print(f"[ASYNC] ERROR: Async target switch failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to restore detection state if needed
        try:
            with g_detection_activation_lock:
                if not g_detection_active:
                    print("[ASYNC] Attempting to restore detection after async switch failure...")
                    g_detection_active = True
        except Exception as restore_error:
            print(f"[ASYNC] Failed to restore detection state: {restore_error}")
    
    finally:
        with g_target_switch_lock:
            g_target_switch_in_progress = False
            g_target_switch_thread = None
        print("[ASYNC] Target switch thread finished and cleaned up")

def start_async_target_switch():
    """Start target switching in a separate thread to avoid blocking video."""
    global g_target_switch_thread, g_target_switch_in_progress
    
    # Check if a target switch is already in progress
    with g_target_switch_lock:
        if g_target_switch_in_progress:
            print("[ASYNC] Target switch already in progress, ignoring new request")
            return False
        
        # Check if previous thread is still alive (shouldn't happen, but safety check)
        if g_target_switch_thread and g_target_switch_thread.is_alive():
            print("[ASYNC] Previous target switch thread still alive, ignoring new request")
            return False
        
        # Start new async target switch thread
        g_target_switch_thread = threading.Thread(
            target=async_target_switch,
            name="AsyncTargetSwitchThread"
        )
        g_target_switch_thread.daemon = True
        g_target_switch_thread.start()
        
        print("[ASYNC] Started async target switch thread")
        return True

def handle_target_switch():
    """Handle target switching by restarting the detection thread."""
    global g_current_target_index, g_target_entities_list, current_detection_thread_handler
    global g_model_object, g_device_object, ONTOLOGY_FOR_VIEWER, latest_detections_for_bbox_send
    global g_last_target_switch_time, g_detection_active
    
    print("=== TARGET SWITCH START (Robust Version) ===")
    
    try:
        with g_order_listener_lock:
            new_target_index = g_current_target_index
        
        print(f"Current target list: {g_target_entities_list}")
        print(f"Requested target index: {new_target_index}")
        
        if new_target_index >= len(g_target_entities_list) or new_target_index == 0:
            print(f"Target switch ignored - invalid index {new_target_index} or no change needed.")
            print(f"Available targets: {len(g_target_entities_list)}")
            return
        
        print(f"Switching target from index 0 to index {new_target_index}...")
        
        # Setup for new target
        current_target_name = g_target_entities_list[new_target_index]
        current_target_caption = f"a {current_target_name}"
        new_ontology_caption = {current_target_name: current_target_caption}
        print(f"New target: '{current_target_name}', caption: '{current_target_caption}'")

        # Store original state for recovery
        original_detection_active = False
        original_ontology = ONTOLOGY_FOR_VIEWER
        
        try:
            # --- STEP 1: Stop current detection thread (NON-BLOCKING) ---
            print("Step 1: Stopping current detection thread...")
            with g_detection_activation_lock:
                original_detection_active = g_detection_active
                g_detection_active = False
            
            print(f"Detection was active: {original_detection_active}")
            
            # Clear detection queues immediately to unblock any waiting operations
            print("Clearing detection queues...")
            cleared_count = 0
            try:
                while not DETECTION_RESULTS_QUEUE.empty(): 
                    try:
                        DETECTION_RESULTS_QUEUE.get_nowait()
                        cleared_count += 1
                    except queue.Empty:
                        break
                
                while not FRAME_FOR_DETECTION_QUEUE.empty():
                    try:
                        FRAME_FOR_DETECTION_QUEUE.get_nowait()
                        cleared_count += 1
                    except queue.Empty:
                        break
                
                print(f"Cleared {cleared_count} items from queues")
                latest_detections_for_bbox_send = None
            except Exception as e:
                print(f"Warning: Error clearing queues: {e}")

            # --- STEP 2: Reload model with new ontology ---
            print("Step 2: Reloading model with new ontology...")
            
            try:
                success, updated_model, updated_device, updated_ontology = update_model_ontology(new_ontology_caption)
                
                if success and updated_model and updated_device and updated_ontology:
                    g_model_object = updated_model
                    g_device_object = updated_device
                    ONTOLOGY_FOR_VIEWER = updated_ontology
                    print(f"✓ Model reloaded for target '{current_target_name}'")
                else:
                    print("✗ Failed to reload model")
                    # Restore original ontology
                    if original_ontology:
                        ONTOLOGY_FOR_VIEWER = original_ontology
                        print("Restored original ontology")
                    # Restore detection state if it was active
                    if original_detection_active:
                        with g_detection_activation_lock:
                            g_detection_active = True
                        print("Restored original detection state")
                    return
            except Exception as e:
                print(f"✗ Error during model reload: {e}")
                # Restore original state
                if original_ontology:
                    ONTOLOGY_FOR_VIEWER = original_ontology
                    print("Restored original ontology")
                if original_detection_active:
                    with g_detection_activation_lock:
                        g_detection_active = True
                    print("Restored original detection state")
                return

            # --- STEP 3: Restart detection thread (IMPROVED) ---
            print("Step 3: Restarting detection thread...")
            
            try:
                # Signal old thread to stop
                DETECTION_THREAD_STOP_EVENT.set()
                
                # Send sentinel to unblock any waiting get() operations
                try:
                    FRAME_FOR_DETECTION_QUEUE.put_nowait(None)
                except queue.Full:
                    # If queue is full, clear it and try again
                    try:
                        FRAME_FOR_DETECTION_QUEUE.get_nowait()
                        FRAME_FOR_DETECTION_QUEUE.put_nowait(None)
                    except queue.Empty:
                        pass
                
                # Wait for old thread to finish (with shorter timeout)
                if current_detection_thread_handler and current_detection_thread_handler.is_alive():
                    print("Waiting for old detection thread to stop...")
                    current_detection_thread_handler.join(timeout=1.0)  # Reduced timeout
                    if current_detection_thread_handler.is_alive():
                        print("Warning: Old detection thread did not stop in time, proceeding anyway")
                
                # Clear the stop event for new thread
                DETECTION_THREAD_STOP_EVENT.clear()
                
                # Start new detection thread
                print("Starting new detection thread...")
                current_detection_thread_handler = threading.Thread(
                    target=detection_worker, 
                    args=(g_model_object, g_device_object, new_ontology_caption),
                    name=f"DetectionWorkerThread-{current_target_name}"
                )
                current_detection_thread_handler.daemon = True
                current_detection_thread_handler.start()
                
                # Small delay to let new thread initialize
                time.sleep(0.1)
                
                print("New detection thread started successfully")
                
            except Exception as e:
                print(f"✗ Error restarting detection thread: {e}")
                # Try to restore original state
                if original_detection_active:
                    try:
                        with g_detection_activation_lock:
                            g_detection_active = True
                        print("Restored detection state after thread restart failure")
                    except:
                        print("Failed to restore detection state")
                return
            
            # --- STEP 4: Activate detection if it was active before ---
            if original_detection_active:
                print("Step 4: Activating detection...")
                try:
                    with g_detection_activation_lock:
                        g_detection_active = True
                    print("✓ Detection reactivated with new target")
                except Exception as e:
                    print(f"✗ Error reactivating detection: {e}")
            else:
                print("Detection remains inactive")
                
            print(f"✓ Target switch completed successfully to '{current_target_name}'")
            return True
                        
        except Exception as e:
            print(f"✗ Error during target switch steps: {e}")
            # Try to restore original state
            try:
                if original_ontology:
                    ONTOLOGY_FOR_VIEWER = original_ontology
                    print("Restored original ontology after error")
                if original_detection_active:
                    with g_detection_activation_lock:
                        g_detection_active = True
                    print("Restored detection state after error")
            except Exception as restore_error:
                print(f"Failed to restore original state: {restore_error}")
            raise  # Re-raise to be caught by outer try-catch
            
    except Exception as e:
        print(f"✗ Critical error during target switch: {e}")
        import traceback
        traceback.print_exc()
    
    print("=== TARGET SWITCH END ===")
    print()

def shutdown_all_threads():
    """Shutdown all threads gracefully."""
    global g_target_switch_thread, g_target_switch_in_progress
    
    print("Shutting down all threads...")
    
    # --- NEW: Stop async target switch thread if running ---
    with g_target_switch_lock:
        if g_target_switch_in_progress and g_target_switch_thread and g_target_switch_thread.is_alive():
            print("Waiting for async target switch thread to complete...")
            # Don't force stop, let it complete gracefully
            
    if g_target_switch_thread and g_target_switch_thread.is_alive():
        print("Joining async target switch thread...")
        g_target_switch_thread.join(timeout=5.0)  # Give it time to complete
        if g_target_switch_thread.is_alive():
            print("Warning: Async target switch thread did not stop in time during shutdown.")
        else:
            print("Async target switch thread stopped successfully")
    # --- END NEW ---
    
    # Deactivate detection
    with g_detection_activation_lock:
        was_active = g_detection_active
        g_detection_active = False
    print(f"Detection was active: {was_active}, now deactivated")
    
    # Set stop events
    DETECTION_THREAD_STOP_EVENT.set()
    print("Detection stop event set")
    
    # Send sentinel to detection queue
    try: 
        FRAME_FOR_DETECTION_QUEUE.put_nowait(None) 
        print("Sentinel sent to detection queue")
    except queue.Full: 
        print("Detection queue was full, sentinel not sent")
    
    # Wait for detection thread
    if current_detection_thread_handler and current_detection_thread_handler.is_alive():
        print("Joining detection thread...")
        current_detection_thread_handler.join(timeout=3.0)  # Increased timeout for shutdown
        if current_detection_thread_handler.is_alive():
            print("Warning: Detection thread did not stop in time during shutdown.")
        else:
            print("Detection thread stopped successfully")
    else:
        print("No active detection thread to stop")
    
    print("Thread shutdown complete.")

if __name__ == '__main__':
    main() 