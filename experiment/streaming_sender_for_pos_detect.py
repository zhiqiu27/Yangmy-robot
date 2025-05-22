#!/usr/bin/env python3
import sys
import pyzed.sl as sl
import argparse
from time import sleep

# Global variable to hold parsed command-line options
opt = None

def parse_args(init_params):
    """
    Parses command-line arguments to set camera resolution.
    """
    if not opt:
        print("[Error] Options not parsed yet.")
        return

    resolution_map = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1200": sl.RESOLUTION.HD1200,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "SVGA": sl.RESOLUTION.SVGA,
        "VGA": sl.RESOLUTION.VGA,
        "AUTO": sl.RESOLUTION.AUTO
    }

    selected_res_key = opt.resolution.upper()
    if selected_res_key in resolution_map:
        init_params.camera_resolution = resolution_map[selected_res_key]
        print(f"[Sample] Using Camera in resolution {selected_res_key}")
    elif len(opt.resolution) > 0 and opt.resolution.upper() != "AUTO":
        print(f"[Sample] Invalid resolution '{opt.resolution}'. Using default (AUTO).")
        init_params.camera_resolution = sl.RESOLUTION.AUTO
    else: # Default or explicit AUTO
        print("[Sample] Using default resolution (AUTO).")
        init_params.camera_resolution = sl.RESOLUTION.AUTO

def main():
    global opt # To access the parsed arguments

    init_params = sl.InitParameters()

    # Apply resolution from command line arguments
    parse_args(init_params)

    # Parameters from pos_detect.py for depth and coordinate system
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.depth_maximum_distance = 10.0  # Max depth for NEURAL mode
    init_params.camera_fps = 15 # You can adjust this, e.g., 10 or 30.

    init_params.sdk_verbose = 1  # Enable verbose SDK logging

    cam = sl.Camera()
    status = cam.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open failed: {status}. Exit program.")
        exit()

    # Runtime parameters (mostly for grab)
    runtime_params = sl.RuntimeParameters()

    # Streaming parameters
    stream_params = sl.StreamingParameters()
    stream_params.codec = sl.STREAMING_CODEC.H264  # H265 (HEVC) is also an option if supported
    stream_params.bitrate = 8000  # Kilobits per second (e.g., 8 Mbps). Adjust as needed for quality/bandwidth.
    stream_params.port = 30000    # Explicitly set streaming port. Make sure this port is open.

    print(f"Attempting to stream on IP: [Your Local IP on this machine] Port: {stream_params.port}")
    print("Note down this IP address and Port to use on the receiver machine.")

    status_streaming = cam.enable_streaming(stream_params)
    if status_streaming != sl.ERROR_CODE.SUCCESS:
        print(f"Streaming initialization error: {status_streaming}")
        cam.close()
        exit()

    print(f"Successfully streaming on port {stream_params.port}. Press Ctrl+C to stop.")

    exit_app = False
    try:
        while not exit_app:
            err = cam.grab(runtime_params)
            if err == sl.ERROR_CODE.SUCCESS:
                # Minimal sleep, the rate is primarily governed by camera FPS and grab()
                sleep(0.001)
            # else:
            #     print(f"Grab error: {err}") # Uncomment for debugging grab issues
            #     sleep(0.01) # Prevent busy-looping on continuous errors
    except KeyboardInterrupt:
        print("\nStopping stream...")
        exit_app = True
    finally:
        # Disable Streaming
        cam.disable_streaming()
        # Close the Camera
        cam.close()
        print("Streaming disabled and camera closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=str,
                        help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA, VGA or AUTO',
                        default='AUTO') # Default to AUTO if not specified
    opt = parser.parse_args() # opt is now globally available for parse_args
    main() 