import cv2
import numpy as np
import socket
import struct # To send/receive data size
import time
import threading # New import
import queue     # New import
from object_detect import initialize_detection_model, run_detection, draw_detections_on_frame

# Global variables
img_display_global = None # Stores the latest raw frame from server for immediate display
latest_detections_for_bbox_send = None # Stores sv.Detections object from detection thread or queue

ONTOLOGY_FOR_VIEWER = None

IMAGE_PORT = 12345
BBOX_PORT = 12346

# Queues for communication between main thread and detection thread
# Queue for frames to be processed by the detection thread
# Maxsize 1: we only care about processing the most recent frame if main loop is faster
FRAME_FOR_DETECTION_QUEUE = queue.Queue(maxsize=1)
# Queue for results (processed_frame, detections_object) from the detection thread
DETECTION_RESULTS_QUEUE = queue.Queue(maxsize=1) # Now stores only sv.Detections objects

DETECTION_THREAD_STOP_EVENT = threading.Event()

def receive_all(sock, n):
    data = bytearray()
    while len(data) < n:
        try:
            # sock.settimeout(1.0) # Setting timeout here for each recv call might be too aggressive if n is large
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

def detection_worker(model, device, ontology):
    """Thread that performs object detection on frames from a queue."""
    print("Detection thread started.")

    # FPS calculation for detection
    detection_count = 0
    last_fps_calc_time = time.time()
    fps_calc_interval = 2 # seconds (calculate FPS every 2 seconds, for example)

    while not DETECTION_THREAD_STOP_EVENT.is_set():
        try:
            raw_frame_to_detect = FRAME_FOR_DETECTION_QUEUE.get(timeout=0.1)
        except queue.Empty:
            continue

        if raw_frame_to_detect is None: 
            DETECTION_RESULTS_QUEUE.put(None) 
            break
        
        # print("Detection worker: Got frame") # DEBUG
        start_detection_time = time.time() # Optional: time individual detections
        detections_obj = run_detection(raw_frame_to_detect, model) 
        # if detections_obj:
        #    print(f"Detection worker: Detection took {time.time() - start_detection_time:.4f}s") # DEBUG

        detection_count += 1 # Increment even if detections_obj is None (means an attempt was made)

        if detections_obj is not None:
            try:
                DETECTION_RESULTS_QUEUE.put_nowait(detections_obj) 
            except queue.Full:
                pass 

        FRAME_FOR_DETECTION_QUEUE.task_done()

        # Calculate and print detection FPS periodically
        current_time = time.time()
        if current_time - last_fps_calc_time >= fps_calc_interval:
            fps = detection_count / (current_time - last_fps_calc_time)
            print(f"[Detection Thread] FPS: {fps:.2f}")
            detection_count = 0
            last_fps_calc_time = current_time

    print("Detection thread finished.")

def main():
    global img_display_global, ONTOLOGY_FOR_VIEWER, latest_detections_for_bbox_send

    print("Initializing detection model...")
    model, device, ontology_obj = initialize_detection_model(model_path="D:/models",ontology_caption={"bottle": "a plastic bottle"})
    if not model:
        print("Failed to initialize detection model. Exiting.")
        return
    ONTOLOGY_FOR_VIEWER = ontology_obj
    print("Detection model initialized.")

    # Start the detection worker thread
    detection_thread = threading.Thread(target=detection_worker, args=(model, device, ONTOLOGY_FOR_VIEWER))
    detection_thread.daemon = True 
    detection_thread.start()

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
        DETECTION_THREAD_STOP_EVENT.set()
        # Send sentinel to ensure detection_worker can exit if blocked on queue.get()
        try: FRAME_FOR_DETECTION_QUEUE.put_nowait(None) 
        except queue.Full: pass
        if detection_thread.is_alive(): detection_thread.join()
        return

    cv2.namedWindow('Object Detection Viewer')
    print("Waiting for images...")

    last_bbox_send_time = time.time()
    bbox_send_interval = 0.1 # Send bbox at most X times per second (e.g., 10 FPS for bbox)

    # FPS calculation for received frames
    received_frame_count = 0
    last_received_fps_time = time.time()
    received_fps_interval = 2 # seconds

    try:
        while True:
            # 1. Receive image (non-blocking for the main display loop as much as possible)
            image_socket.settimeout(1.0) # Shorter timeout for recv to keep UI responsive
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
                        img_display_global = decoded_frame # Update for immediate display
                        received_frame_count += 1 # Increment for FPS calculation
                        
                        # Try to put the new frame into the detection queue without blocking main thread
                        try:
                            FRAME_FOR_DETECTION_QUEUE.put_nowait(img_display_global.copy()) # Send a copy
                        except queue.Full:
                            # print("Frame for detection queue is full. Skipping this frame for detection.")
                            pass # Detection thread is busy, skip this frame for detection
                    else:
                        print("Failed to decode image.")
                except Exception as e:
                    print(f"Error decoding image: {e}")
                    # continue or break depending on severity

            # --- Calculate and Display Received Frames FPS ---
            current_time_fps_calc = time.time()
            if current_time_fps_calc - last_received_fps_time >= received_fps_interval:
                fps = received_frame_count / (current_time_fps_calc - last_received_fps_time)
                print(f"[Main Thread - Image Reception] FPS: {fps:.2f}")
                received_frame_count = 0
                last_received_fps_time = current_time_fps_calc

            # --- Display Logic (MODIFIED) ---
            display_frame_final = None
            if img_display_global is not None:
                display_frame_final = img_display_global.copy() # Always start with a copy of the raw frame

                # Check for new detection results from the queue
                new_detections_from_queue = None
                try:
                    new_detections_from_queue = DETECTION_RESULTS_QUEUE.get_nowait()
                    if new_detections_from_queue is None and DETECTION_THREAD_STOP_EVENT.is_set():
                        # Sentinel from detection thread means it's stopping, might be an error or shutdown
                        print("Received stop sentinel from detection results queue.")
                        # We might want to stop the main loop if this happens unexpectedly
                        # For now, just ensures latest_detections_for_bbox_send is not set to None if it was valid
                    elif new_detections_from_queue is not None: # Valid detections object
                        latest_detections_for_bbox_send = new_detections_from_queue
                    # If queue empty, latest_detections_for_bbox_send remains as is (previous valid detection or None)
                except queue.Empty:
                    pass 

                # If we have any valid detections (either new or last known), draw them
                if latest_detections_for_bbox_send: # Ensure it's not None
                    # Ensure ONTOLOGY_FOR_VIEWER is available
                    if ONTOLOGY_FOR_VIEWER is not None:
                        draw_detections_on_frame(display_frame_final, latest_detections_for_bbox_send, ONTOLOGY_FOR_VIEWER)
                    else:
                        print("Warning: ONTOLOGY_FOR_VIEWER not set, cannot draw detections.")
            else:
                display_frame_final = img_display_global # Fallback to raw image if display_frame_final somehow became None
            
            if display_frame_final is not None:
                cv2.imshow('Object Detection Viewer', display_frame_final)
            
            # --- Bbox Sending Logic --
            current_time = time.time()
            if current_time - last_bbox_send_time > bbox_send_interval:
                # Bbox sending now relies on latest_detections_for_bbox_send, which is updated by the display logic
                if latest_detections_for_bbox_send and \
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
                        print("Timeout sending bbox on BBOX_SOCKET. Connection issue?")
                        # break # Decide if break is too drastic
                    except socket.error as e:
                        print(f"Error sending bbox on BBOX_SOCKET: {e}")
                        # break
                else:
                    try:
                        bbox_socket.settimeout(0.5)
                        bbox_socket.sendall(struct.pack('>I', 0))
                        last_bbox_send_time = current_time
                    except socket.timeout:
                        print("Timeout sending zero-size bbox signal on BBOX_SOCKET.")
                        # break
                    except socket.error as e:
                        print(f"Error sending zero-size bbox on BBOX_SOCKET: {e}")
                        # break

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
        DETECTION_THREAD_STOP_EVENT.set()
        try: FRAME_FOR_DETECTION_QUEUE.put_nowait(None) # Ensure detection_worker can exit queue.get()
        except queue.Full: pass
        # The detection_worker itself will put None in DETECTION_RESULTS_QUEUE upon seeing stop event + sentinel
        
        if detection_thread.is_alive():
            print("Joining detection thread...")
            detection_thread.join(timeout=2.0)
            if detection_thread.is_alive():
                print("Detection thread did not stop in time.")

        if image_socket: image_socket.close()
        if bbox_socket: bbox_socket.close()
        cv2.destroyAllWindows()
        print("Client shut down complete.")

if __name__ == '__main__':
    main() 