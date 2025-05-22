import time
import math
import threading
from unitree_sdk2py.go2.sport.sport_client import SportClient # For type hinting

class PitchController:
    def __init__(self, sport_client: SportClient, interpolation_duration_s: float = 2.0):
        self.sport_client = sport_client
        self.fixed_roll_rad = 0.0  # Roll is always 0

        self._shared_state = {
            "final_target_pitch_rad": 0.0,
            "interpolation_start_pitch_rad": 0.0,
            "interpolation_start_time_ns": 0,
            "interpolation_duration_s": interpolation_duration_s,
            "current_sent_pitch_rad": 0.0,
            "send_euler_active": False,
            "final_target_yaw_rad": 0.0,  # Fixed at 0 for this controller
        }
        self._state_lock = threading.Lock()
        self._keep_sending_event = threading.Event()
        self._cmd_thread = None
        print(f"PitchController initialized. Interpolation duration: {interpolation_duration_s}s.")

    def set_current_yaw_as_target(self, current_robot_yaw_rad: float):
        with self._state_lock:
            self._shared_state["final_target_yaw_rad"] = current_robot_yaw_rad
            # print(f"PitchController: Updated target base yaw to {math.degrees(current_robot_yaw_rad):.1f}°") # Debug

    def _euler_command_thread_func(self):
        print("PitchController: Euler command thread started (pitch control only, roll fixed at 0).")
        while not self._keep_sending_event.is_set():
            pitch_to_send = 0.0
            yaw_to_send = 0.0 
            send_command_now = False

            with self._state_lock:
                if self._shared_state["send_euler_active"]:
                    send_command_now = True
                    now_ns = time.time_ns()
                    elapsed_s = (now_ns - self._shared_state["interpolation_start_time_ns"]) / 1e9
                    duration_s = self._shared_state["interpolation_duration_s"]

                    start_pitch = self._shared_state["interpolation_start_pitch_rad"]
                    final_pitch = self._shared_state["final_target_pitch_rad"]
                    yaw_to_send = self._shared_state["final_target_yaw_rad"]

                    if elapsed_s < duration_s and duration_s > 0:
                        progress = elapsed_s / duration_s
                        pitch_to_send = start_pitch + (final_pitch - start_pitch) * progress
                    else:
                        pitch_to_send = final_pitch
                    
                    self._shared_state["current_sent_pitch_rad"] = pitch_to_send
                else:
                    # If not active, ensure values are from a known state (e.g. last sent)
                    pitch_to_send = self._shared_state["current_sent_pitch_rad"]
                    yaw_to_send = self._shared_state["final_target_yaw_rad"]

            if send_command_now:
                #print(f"[PitchController] 发送 pitch={math.degrees(pitch_to_send):.2f}° yaw={math.degrees(yaw_to_send):.2f}°")
                self.sport_client.Euler(self.fixed_roll_rad, pitch_to_send, 0)
            
            time.sleep(0.05)  # Approx 20Hz
        print("PitchController: Euler command thread stopped.")

    def start_control(self):
        if self._cmd_thread is not None and self._cmd_thread.is_alive():
            print("PitchController: Control is already active.")
            return

        print("PitchController: Starting pitch control.")
        self._keep_sending_event.clear()
        with self._state_lock:
            # Initialize to current pitch (or 0 if first time after creation)
            # This ensures that if start_control is called, it maintains current pitch or starts from 0.
            self._shared_state["interpolation_start_pitch_rad"] = self._shared_state["current_sent_pitch_rad"]
            self._shared_state["final_target_pitch_rad"] = self._shared_state["current_sent_pitch_rad"] 
            self._shared_state["interpolation_start_time_ns"] = time.time_ns()
            self._shared_state["send_euler_active"] = True
        
        self._cmd_thread = threading.Thread(target=self._euler_command_thread_func, daemon=True)
        self._cmd_thread.start()

    def stop_control(self, transition_to_zero_pitch: bool = True):
        if self._cmd_thread is None or not self._cmd_thread.is_alive():
            print("PitchController: Control is not active or already stopped.")
            # Ensure state reflects inactivity if called when not running
            with self._state_lock:
                 self._shared_state["send_euler_active"] = False
            self._keep_sending_event.set() # Signal just in case thread is somehow stuck without being None
            return

        print("PitchController: Stopping pitch control...")
        if transition_to_zero_pitch and self._shared_state["send_euler_active"]:
            duration = self._shared_state['interpolation_duration_s']
            print(f"PitchController: Transitioning pitch to 0° over {duration}s...")
            with self._state_lock:
                self._shared_state["interpolation_start_pitch_rad"] = self._shared_state["current_sent_pitch_rad"]
                self._shared_state["final_target_pitch_rad"] = 0.0
                self._shared_state["interpolation_start_time_ns"] = time.time_ns()
                # send_euler_active remains true for this transition
            
            time.sleep(duration + 0.1) # Wait for transition to complete
            print("PitchController: Transition to 0 pitch should be complete.")

        with self._state_lock:
            self._shared_state["send_euler_active"] = False
        self._keep_sending_event.set()
        
        if self._cmd_thread.is_alive(): # Check again after setting event
            print("PitchController: Waiting for command thread to finish...")
            self._cmd_thread.join(timeout=1.5)
            if self._cmd_thread.is_alive():
                print("PitchController: Warning: Command thread did not finish in time.")
        self._cmd_thread = None
        print("PitchController: Control stopped.")

    def set_pitch(self, pitch_degrees: float, desired_target_yaw_rad: float = None):
        if self._cmd_thread is None or not self._cmd_thread.is_alive() or not self._shared_state["send_euler_active"]:
            print(f"[PitchController] set_pitch({pitch_degrees}): 线程未启动或未激活！")
            print(f"[PitchController] _cmd_thread is None: {self._cmd_thread is None}, is_alive: {self._cmd_thread.is_alive() if self._cmd_thread else 'N/A'}, send_euler_active: {self._shared_state['send_euler_active']}")
            print("PitchController: Control not active. Call start_control() first to set pitch.")
            return

        pitch_radians = math.radians(pitch_degrees)
        duration = self._shared_state['interpolation_duration_s']
        print(f"[PitchController] New target pitch: {pitch_degrees}° ({pitch_radians:.3f} rad). Interpolating over {duration}s...")
        with self._state_lock:
            self._shared_state["interpolation_start_pitch_rad"] = self._shared_state["current_sent_pitch_rad"]
            self._shared_state["final_target_pitch_rad"] = pitch_radians
            self._shared_state["interpolation_start_time_ns"] = time.time_ns()
            if desired_target_yaw_rad is not None:
                self._shared_state["final_target_yaw_rad"] = desired_target_yaw_rad

    def reset_pitch(self, desired_target_yaw_rad: float = None):
        print("PitchController: Resetting pitch to 0°.")
        self.set_pitch(0.0, desired_target_yaw_rad=desired_target_yaw_rad)

# No if __name__ == "__main__": block here, this file is now a module. 