import cv2
import numpy as np
import socket
import struct # To send/receive data size
import time

# Global variables for mouse callback
drawing = False
ix, iy = -1, -1
bbox_to_send = None # Renamed from bbox to avoid confusion with local drawing bbox
img_display_global = None # To store the latest full image for drawing

# Mouse callback function to draw a rectangle
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, bbox_to_send, img_display_global

    if img_display_global is None: # Don't do anything if there's no image
        return

    img_copy_for_draw = img_display_global.copy() # Work on a copy of the latest full image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        bbox_to_send = None # Reset bbox when starting to draw a new one

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(img_copy_for_draw, (ix, iy), (x, y), (0, 255, 0), 1) # Draw dynamically
            cv2.imshow('Received Image - Draw BBox', img_copy_for_draw)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Ensure x1 < x2 and y1 < y2
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        
        # Prevent zero-width or zero-height boxes
        if x1 == x2 or y1 == y2:
            print("Bounding box has zero width or height. Please redraw.")
            bbox_to_send = None
            cv2.imshow('Received Image - Draw BBox', img_display_global) # Show original if bbox is invalid
        else:
            bbox_to_send = (x1, y1, x2, y2)
            cv2.rectangle(img_copy_for_draw, (bbox_to_send[0], bbox_to_send[1]), (bbox_to_send[2], bbox_to_send[3]), (0, 0, 255), 2) # Draw final in red
            cv2.imshow('Received Image - Draw BBox', img_copy_for_draw)
            print(f"Bounding box selected: {bbox_to_send}")

def receive_all(sock, n):
    # Helper function to receive n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def main():
    global bbox_to_send, img_display_global

    # --- Network Setup ---
    host_ip_default = '192.168.123.15' # Example, common for Unitree Go2
    host_ip_input = input(f"Enter Go2's IP address (default: {host_ip_default}): ")
    host_ip = host_ip_input if host_ip_input else host_ip_default
    port = 12345

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print(f"Attempting to connect to server at {host_ip}:{port}...")
        client_socket.connect((host_ip, port))
        print(f"Successfully connected to server.")
    except ConnectionRefusedError:
        print(f"Connection refused. Ensure the server script is running on Go2 ({host_ip}:{port}).")
        return
    except socket.timeout:
        print(f"Connection timed out. Ensure Go2 is reachable and server is running.")
        return
    except socket.gaierror:
        print(f"Hostname could not be resolved. Check the IP address: {host_ip}")
        return
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return

    cv2.namedWindow('Received Image - Draw BBox')
    cv2.setMouseCallback('Received Image - Draw BBox', draw_rectangle)

    print("\\nInstructions:")
    print(" - Draw a rectangle on the image with your mouse (left-click and drag).")
    print(" - Press 's' to send the current bounding box to Go2.")
    print(" - Press 'c' to clear the current bounding box selection.")
    print(" - Press 'q' to quit.")
    print("Waiting for images from Go2...")

    try:
        while True:
            # 1. Receive image size (as 4-byte big-endian unsigned integer)
            img_size_data = receive_all(client_socket, 4)
            if not img_size_data:
                print("Server closed connection (failed to get image size).")
                break
            img_size = struct.unpack('>I', img_size_data)[0]

            if img_size == 0:
                # This could be a signal from the server that it's alive but has no image,
                # or a specific signal to wait. For now, we'll just continue.
                # print("Received zero image size signal from server.")
                time.sleep(0.01) # Avoid busy-looping if server sends many 0-size packets
                continue

            # 2. Receive image data
            img_data_jpeg = receive_all(client_socket, img_size)
            if not img_data_jpeg:
                print("Server closed connection (failed to get image data).")
                break

            # 3. Decode and display image
            try:
                img_np_arr = np.frombuffer(img_data_jpeg, dtype=np.uint8)
                img_display_global = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)

                if img_display_global is None:
                    print("Failed to decode image. Skipping frame.")
                    continue
                
                # Create a working copy for display to draw on, if bbox exists
                display_frame = img_display_global.copy()
                if bbox_to_send: # If a bbox was previously drawn and not cleared, show it
                     cv2.rectangle(display_frame, (bbox_to_send[0], bbox_to_send[1]), (bbox_to_send[2], bbox_to_send[3]), (0, 0, 255), 2)
                cv2.imshow('Received Image - Draw BBox', display_frame)

            except Exception as e:
                print(f"Error decoding or displaying image: {e}")
                continue # Try to recover by getting the next frame

            key = cv2.waitKey(1) & 0xFF # Crucial for OpenCV GUI to update
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                if bbox_to_send:
                    try:
                        # Send bounding box: x1,y1,x2,y2 as a string
                        bbox_str = f"{bbox_to_send[0]},{bbox_to_send[1]},{bbox_to_send[2]},{bbox_to_send[3]}"
                        bbox_payload = bbox_str.encode('utf-8')
                        
                        # Send size of bbox_payload first (4-byte big-endian unsigned int)
                        client_socket.sendall(struct.pack('>I', len(bbox_payload)))
                        # Send bbox_payload
                        client_socket.sendall(bbox_payload)
                        print(f"Sent bounding box: {bbox_str}")
                        # Optionally clear bbox_to_send after sending, or keep it for re-sending
                        # bbox_to_send = None 
                        # cv2.imshow('Received Image - Draw BBox', img_display_global) # Refresh display
                    except socket.error as e:
                        print(f"Socket error while sending bbox: {e}. Assuming connection is broken.")
                        break 
                    except Exception as e:
                        print(f"Error sending bounding box: {e}")
                else:
                    print("No bounding box selected to send. Draw one first or press 'c' to clear if one is stuck.")
            elif key == ord('c'):
                bbox_to_send = None
                if img_display_global is not None: # Refresh display without bbox
                    cv2.imshow('Received Image - Draw BBox', img_display_global)
                print("Bounding box cleared.")

    except KeyboardInterrupt:
        print("\\nInterrupted by user (Ctrl+C).")
    except socket.error as e:
        print(f"Socket error during communication: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing client socket and windows.")
        client_socket.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 