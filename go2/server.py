import socket
import struct
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Server configuration
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 12345      # Same as original image port
SEND_TIMEOUT = 5.0  # Timeout for sending data (seconds)
FRAME_INTERVAL = 1.0 / 24  # Simulate 24 FPS (41.67ms per frame)
PAYLOAD_SIZE = 1000  # Fixed dummy payload size (bytes)

def send_all(sock, data, timeout=SEND_TIMEOUT):
    """Send data with timeout."""
    original_timeout = sock.gettimeout()
    try:
        sock.settimeout(timeout)
        sock.sendall(data)
        return True
    except socket.timeout:
        logger.warning(f"Timeout during sendall after {timeout}s")
        return False
    except socket.error as e:
        logger.error(f"Socket error during send: {e}")
        return False
    finally:
        sock.settimeout(original_timeout)

def main():
    logger.info(f"Starting test server on {HOST}:{PORT}")
    
    # Create server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    server_socket.settimeout(5.0)  # Timeout for accepting connections

    try:
        while True:
            logger.info("Waiting for client connection...")
            try:
                conn, addr = server_socket.accept()
                logger.info(f"Client connected from {addr}")
                # Set larger send buffer
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB
                logger.info(f"Send buffer size: {conn.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)} bytes")
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Error accepting connection: {e}")
                break

            try:
                frame_count = 0
                while True:
                    # Create dummy payload
                    pseudo_data = b'\x00' * PAYLOAD_SIZE
                    size = len(pseudo_data)
                    
                    # Send 4-byte size field
                    start_time = time.time()
                    if not send_all(conn, struct.pack('!I', size)):
                        logger.error("Failed to send size field")
                        break
                    
                    # Send payload
                    if not send_all(conn, pseudo_data):
                        logger.error("Failed to send payload")
                        break
                    
                    frame_count += 1
                    elapsed = time.time() - start_time
                    logger.info(f"Sent frame {frame_count}, size: {size} bytes, took {elapsed:.3f}s")
                    
                    # Simulate frame rate
                    time.sleep(max(0, FRAME_INTERVAL - elapsed))
            
            except Exception as e:
                logger.error(f"Connection error: {e}")
            finally:
                logger.info("Closing client connection")
                try:
                    conn.close()
                except:
                    pass
    
    except KeyboardInterrupt:
        logger.info("Server interrupted")
    finally:
        logger.info("Shutting down server")
        server_socket.close()

if __name__ == "__main__":
    main()
