import cv2
import numpy as np
import json
import serial
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from threading import Thread, Lock
import queue

# --- Ball detection using calibration ---
def detect_ball_x(frame, lower_hsv, upper_hsv):
    """Detect ball in frame and return normalized x-position and visualization frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, frame
    largest = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(largest)
    vis_frame = frame.copy()
    if radius > 5:
        cv2.circle(vis_frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
        cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        normalized_x = x / frame.shape[1]  # 0-1
        return True, normalized_x, vis_frame
    return False, None, vis_frame

# --- PID Controller ---
class BasicPIDController:
    def __init__(self, config_file="config.json"):
        """Initialize controller using calibration file."""
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        # PID gains
        self.Kp = 1.1
        self.Ki = 0.9
        self.Kd = 1.1

        # Scale factor
        self.scale_factor = self.config['calibration']['pixel_to_meter_ratio'] * self.config['camera']['frame_width']

        # Servo info
        self.servo_port = self.config['servo A']['port']
        # self.servo_neutral_angles = [
        #     self.config['servo A']['neutral_angle'],
        #     self.config['servo B']['neutral_angle'],
        #     self.config['servo C']['neutral_angle']
        # ]

        self.servo_neutral_angles = [
          40,
          40,
          40
        ]
        
        self.last_angles = self.servo_neutral_angles.copy()
        self.servo = None
        
        # Active motor selection (0=A, 1=B, 2=C)
        self.active_motor = 0  # Default to motor A

        # Ball tracking
        self.setpoint = 0.0
        self.integral = 0.0
        self.prev_error = 0.0

        # Logs
        self.time_log = []
        self.position_log = []
        self.setpoint_log = []
        self.control_log = []
        self.start_time = None

        # Thread-safe queues
        self.position_queue = queue.Queue(maxsize=1)
        self.display_queue = queue.Queue(maxsize=1)
        self.running = False
        self.camera_ready = False
        self.pid_lock = Lock()

        # Load HSV bounds
        self.lower_hsv = self.config['ball_detection']['lower_hsv']
        self.upper_hsv = self.config['ball_detection']['upper_hsv']

        # Platform center from calibration (centroid of 3 peg points)
        calibration = self.config.get('calibration', {})
        platform_center = calibration.get('platform_center')
        if platform_center:
            self.platform_center = (platform_center[0], platform_center[1])
        else:
            # Fallback: use frame center if no platform center is defined
            self.platform_center = (self.config['camera']['frame_width'] / 2, 
                                   self.config['camera']['frame_height'] / 2)
            print("[WARN] No platform center in config, using frame center as fallback")
        
        # Load 3D peg points for visualization (optional)
        self.peg_points_3d = calibration.get('peg_points_3d')
        
        # Legacy: keep lineA for backward compatibility (used for visualization)
        self.lineA = self.config['servo A'].get('peg_points', [(0,0),(320,0)])  # fallback

    # --- Servo Communication ---
    def connect_servo(self):
        try:
            self.servo = serial.Serial(self.servo_port, 9600)
            time.sleep(2)
            print("[SERVO] Connected")
            return True
        except Exception as e:
            print(f"[SERVO] Failed: {e}")
            return False

    def send_servo_angle(self, output):
        """Send control output to active motor; keep other motors at neutral."""
        # Get active motor (thread-safe)
        with self.pid_lock:
            active_motor = self.active_motor
        
        # Reset all angles to neutral
        self.last_angles = self.servo_neutral_angles.copy()
        
        # Apply control output only to the active motor (output is already absolute angle 0-120)
        self.last_angles[active_motor] = int(np.clip(output, 0, 120))
        
        cmd = f"{self.last_angles[0]} {self.last_angles[1]} {self.last_angles[2]}\n"
        print(cmd)
        try:
            self.servo.write(cmd.encode())
        except Exception as e:
            print(f"[SERVO] Send failed: {e}")

    # --- PID ---
    def update_pid(self, position, dt=0.033):
        with self.pid_lock:
            kp = self.Kp
            ki = self.Ki
            kd = self.Kd
            setpoint = self.setpoint

        error = position - setpoint
        # error *= 100  # scale
        P = kp * error
        self.integral += error * dt
        I = ki * self.integral
        derivative = (error - self.prev_error) / dt
        D = kd * derivative
        self.prev_error = error
        
        # Calculate raw PID output
        pid_output = P + I + D
        
        # Get active motor to use motor-specific thresholds
        with self.pid_lock:
            active_motor = self.active_motor
        
        # Per-motor sensitivity scaling factors (to compensate for different motor sensitivities)
        # Values < 1.0 reduce sensitivity, > 1.0 increase sensitivity
        # Motor B is more sensitive, so scale it down
        motor_sensitivity_scales = [1.0, 1.0, 1.0]  # [Motor A, Motor B, Motor C]
        sensitivity_scale = motor_sensitivity_scales[active_motor]
        
        # Scale PID output based on motor sensitivity
        pid_output = pid_output * sensitivity_scale
        
        # Center output around motor's neutral angle (40 degrees)
        # This gives us -40 to +80 range from neutral, which maps to 0-120 servo range
        center_angle = 40.0  # Motor's physical neutral position
        output = center_angle + pid_output
        
        # Apply minimum output threshold to overcome friction/inertia near center
        # When error is small, PID output is tiny and motor may not move
        # This ensures minimum movement to overcome static friction
        # Per-motor thresholds (can be adjusted individually if needed)
        min_output_thresholds = [5, 0, 5]  # [Motor A, Motor B, Motor C] - Minimum degrees to move from center
        min_output_threshold = min_output_thresholds[active_motor]
        
        # Extra threshold for the side that needs more support (away from motor)
        # direction = 1: error >= 0 (one side), direction = -1: error < 0 (other side)
        # Negative error side (direction == -1) struggles more, so needs extra threshold
        # Per-motor extra thresholds (can be adjusted individually if needed)
        extra_thresholds_negative_error_side = [10, 10, 10]  # [Motor A, Motor B, Motor C] - Extra degrees for negative error side
        extra_threshold_negative_error_side = extra_thresholds_negative_error_side[active_motor]
        if abs(pid_output) < min_output_threshold and abs(error) > 0.0005:
            # Apply minimum movement in direction of error
            direction = 1 if pid_output >= 0 else -1
            if direction == 1:
                # Ball on positive error side (normal threshold)
                output = center_angle + min_output_threshold 
            else:
                # Ball on negative error side - add extra support since this side struggles more
                output = center_angle - min_output_threshold - extra_threshold_negative_error_side
        
        # Clip to valid servo range (0-120 degrees)
        output = np.clip(output, 0, 120)
        print("P: ", P, "kp: ", kp, "Error: ", error, "PID_out: ", pid_output, "Output: ", output)
        # print(f"[PID] Error: {error:.3f}, P: {P:.1f}, I: {I:.1f}, D: {D:.1f}, Output: {output:.1f}")
        return output

    def calculate_motor_position(self, ball_x_norm, center_x, center_y, frame_width, frame_height, active_motor):
        """Calculate ball position along the active motor's axis.
        
        Projects the ball position onto the line from platform center 
        to the active motor's peg point.
        
        Args:
            ball_x_norm: Normalized ball x position (0-1)
            center_x, center_y: Platform center coordinates
            frame_width, frame_height: Frame dimensions in pixels
            active_motor: Motor index (0=A, 1=B, 2=C)
        
        Returns:
            position_m: Ball position in meters along the motor's axis
        """
        # Convert normalized x to pixel coordinates
        # Note: detect_ball_x only returns x_norm, so we estimate y at center for now
        ball_x_pixel = ball_x_norm * frame_width
        ball_y_pixel = center_y  # Assume ball is at center y (single-axis control)
        
        # Get ball offset from center
        dx = ball_x_pixel - center_x
        dy = ball_y_pixel - center_y  # Will be 0 for now, but kept for future 2D support
        
        # If we have peg points, project onto the active motor's axis
        if self.peg_points_3d and len(self.peg_points_3d) == 3:
            # Get the active motor's peg point
            # Frame is resized to calibration size, so no scaling needed
            peg_point = self.peg_points_3d[active_motor]
            peg_x = peg_point[0]
            peg_y = peg_point[1]
            
            # Calculate direction vector from center to peg point
            dir_x = peg_x - center_x
            dir_y = peg_y - center_y
            dir_length = np.sqrt(dir_x**2 + dir_y**2)
            
            if dir_length > 0:
                # Normalize direction vector
                dir_x /= dir_length
                dir_y /= dir_length
                
                # Project ball position onto this direction
                # Dot product of (dx, dy) with normalized direction vector
                projection = dx * dir_x + dy * dir_y
            else:
                # Fallback: use x-axis projection
                projection = dx
        else:
            # Fallback: use x-axis for all motors
            projection = dx
        
        # Convert to meters
        position_m = projection * self.scale_factor
        return position_m
 
    # --- Camera Thread ---
    def camera_thread(self):
        cap = cv2.VideoCapture(self.config['camera']['index'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['frame_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['frame_height'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Verify actual captured frame size (camera may not honor requested resolution)
        ret, test_frame = cap.read()
        if ret:
            actual_w, actual_h = test_frame.shape[1], test_frame.shape[0]
            requested_w = self.config['camera']['frame_width']
            requested_h = self.config['camera']['frame_height']
            if actual_w != requested_w or actual_h != requested_h:
                print(f"[CAMERA] Requested {requested_w}x{requested_h}, but camera captured {actual_w}x{actual_h}")
                print(f"[CAMERA] Resizing to {requested_w}x{requested_h} to match calibration dimensions")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Resize frame to match calibration dimensions to ensure peg points align correctly
            # The camera may not capture at exactly the requested resolution, so we force it
            frame = cv2.resize(frame, (self.config['camera']['frame_width'], 
                                       self.config['camera']['frame_height']))
            
            found, x_norm, vis_frame = detect_ball_x(frame, self.lower_hsv, self.upper_hsv)
            
            # Use platform center from calibration (centroid of 3 peg points)
            # Frame is now exactly the calibration size, so no scaling needed
            mid_x = self.platform_center[0]
            mid_y = self.platform_center[1]
            
            if found:
                # Get active motor (thread-safe)
                with self.pid_lock:
                    active_motor = self.active_motor
                
                # Calculate ball position along the active motor's axis
                position_m = self.calculate_motor_position(
                    x_norm, mid_x, mid_y, frame.shape[1], frame.shape[0], active_motor)
                
                try:
                    if self.position_queue.full():
                        self.position_queue.get_nowait()
                    self.position_queue.put_nowait(position_m)
                except Exception:
                    pass
            
            # Draw platform center marker
            center_pixel = (int(mid_x), int(mid_y))
            cv2.circle(vis_frame, center_pixel, 5, (255, 0, 255), -1)  # Magenta dot
            cv2.circle(vis_frame, center_pixel, 10, (255, 0, 255), 1)  # Magenta ring
            
            # Draw 3D peg points if available (frame is now calibration size, no scaling needed)
            if self.peg_points_3d:
                servo_names = ["A", "B", "C"]
                colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]  # Green, Yellow, Cyan
                for i, peg in enumerate(self.peg_points_3d):
                    peg_point = (int(peg[0]), int(peg[1]))
                    cv2.circle(vis_frame, peg_point, 3, colors[i], -1)
                # Draw triangle connecting peg points
                if len(self.peg_points_3d) == 3:
                    pts = [(int(p[0]), int(p[1])) for p in self.peg_points_3d]
                    cv2.line(vis_frame, pts[0], pts[1], (128, 128, 128), 1)
                    cv2.line(vis_frame, pts[1], pts[2], (128, 128, 128), 1)
                    cv2.line(vis_frame, pts[2], pts[0], (128, 128, 128), 1)
            
            # Legacy: draw lineA if it exists (for backward compatibility)
            if self.lineA and len(self.lineA) == 2:
                lineA_points = [(int(p[0]), int(p[1])) for p in self.lineA]
                cv2.line(vis_frame, lineA_points[0], lineA_points[1], (255, 0, 0), 1)
            
            # Draw target setpoint marker
            with self.pid_lock:
                setpoint = self.setpoint
                active_motor = self.active_motor
            if self.scale_factor != 0:
                # Calculate target position along the active motor's axis
                target_offset_meters = setpoint
                target_offset_pixels = target_offset_meters / self.scale_factor
                
                # Project target onto the active motor's axis (same as ball position)
                if self.peg_points_3d and len(self.peg_points_3d) == 3:
                    # Get the active motor's peg point (frame is calibration size, no scaling needed)
                    peg_point = self.peg_points_3d[active_motor]
                    peg_x = peg_point[0]
                    peg_y = peg_point[1]
                    
                    # Calculate direction vector from center to peg point
                    dir_x = peg_x - mid_x
                    dir_y = peg_y - mid_y
                    dir_length = np.sqrt(dir_x**2 + dir_y**2)
                    
                    if dir_length > 0:
                        # Normalize direction vector
                        dir_x /= dir_length
                        dir_y /= dir_length
                        
                        # Position target along this direction
                        target_x = int(mid_x + target_offset_pixels * dir_x)
                        target_y = int(mid_y + target_offset_pixels * dir_y)
                    else:
                        # Fallback: use x-axis
                        target_x = int(mid_x + target_offset_pixels)
                        target_y = int(mid_y)
                else:
                    # Fallback: use x-axis
                    target_x = int(mid_x + target_offset_pixels)
                    target_y = int(mid_y)
                
                # Ensure target is within frame bounds
                target_x = np.clip(target_x, 0, frame.shape[1] - 1)
                target_y = np.clip(target_y, 0, frame.shape[0] - 1)
                
                # Draw a cross marker at target position
                marker_size = 15
                cv2.line(vis_frame, 
                        (target_x - marker_size, target_y), 
                        (target_x + marker_size, target_y), 
                        (0, 255, 255), 3)  # Yellow horizontal line
                cv2.line(vis_frame, 
                        (target_x, target_y - marker_size), 
                        (target_x, target_y + marker_size), 
                        (0, 255, 255), 3)  # Yellow vertical line
                # Draw a circle around the target
                cv2.circle(vis_frame, (target_x, target_y), marker_size + 5, (0, 255, 255), 2)
                # Label the target with setpoint value
                cv2.putText(vis_frame, f"Target: {setpoint:.3f}m", 
                           (target_x + marker_size + 5, target_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            try:
                if self.display_queue.full():
                    self.display_queue.get_nowait()
                self.display_queue.put_nowait(vis_frame)
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()

    # --- Control Thread ---
    def control_thread(self):
        if not self.connect_servo():
            print("[ERROR] Servo not connected - simulation mode")
        self.start_time = time.time()
        while self.running:
            try:
                position = self.position_queue.get(timeout=0.1)
                control_output = self.update_pid(position)
                self.send_servo_angle(control_output)
                current_time = time.time() - self.start_time
                self.time_log.append(current_time)
                self.position_log.append(position)
                self.setpoint_log.append(self.setpoint)
                self.control_log.append(control_output)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[CONTROL] Error: {e}")
                break

    # --- GUI ---
    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("PID Controller")
        self.root.geometry("520x400")

        ttk.Label(self.root, text="PID Gains", font=("Arial", 18, "bold")).pack(pady=10)

        # Kp
        ttk.Label(self.root, text="Kp").pack()
        self.kp_var = tk.DoubleVar(value=self.Kp)
        kp_slider = ttk.Scale(self.root, from_=0, to=100, variable=self.kp_var, orient=tk.HORIZONTAL, length=500)
        kp_slider.pack()
        self.kp_label = ttk.Label(self.root, text=f"Kp: {self.Kp:.1f}")
        self.kp_label.pack()

        # Ki
        ttk.Label(self.root, text="Ki").pack()
        self.ki_var = tk.DoubleVar(value=self.Ki)
        ki_slider = ttk.Scale(self.root, from_=0, to=10, variable=self.ki_var, orient=tk.HORIZONTAL, length=500)
        ki_slider.pack()
        self.ki_label = ttk.Label(self.root, text=f"Ki: {self.Ki:.1f}")
        self.ki_label.pack()

        # Kd
        ttk.Label(self.root, text="Kd").pack()
        self.kd_var = tk.DoubleVar(value=self.Kd)
        kd_slider = ttk.Scale(self.root, from_=0, to=20, variable=self.kd_var, orient=tk.HORIZONTAL, length=500)
        kd_slider.pack()
        self.kd_label = ttk.Label(self.root, text=f"Kd: {self.Kd:.1f}")
        self.kd_label.pack()

        # Setpoint slider
        ttk.Label(self.root, text="Setpoint (m)").pack()
        # Use beam length to determine setpoint range (limits were removed from calibration)
        beam_length = self.config.get('beam_length_m', 0.30)
        # Check if limits exist in config (for backward compatibility)
        servo_a_config = self.config.get('servo A', {})
        pos_min = servo_a_config.get('position_min_m')
        pos_max = servo_a_config.get('position_max_m')
        
        # Use beam length-based defaults if limits not available
        if pos_min is None:
            pos_min = -beam_length / 2
        if pos_max is None:
            pos_max = beam_length / 2
            
        self.setpoint_var = tk.DoubleVar(value=self.setpoint)
        sp_slider = ttk.Scale(self.root, from_=pos_min, to=pos_max, variable=self.setpoint_var, orient=tk.HORIZONTAL, length=500)
        sp_slider.pack()
        self.setpoint_label = ttk.Label(self.root, text=f"Setpoint: {self.setpoint:.3f}")
        self.setpoint_label.pack()

        # Motor selection buttons
        ttk.Label(self.root, text="Active Motor", font=("Arial", 12, "bold")).pack(pady=(10, 5))
        motor_frame = tk.Frame(self.root)
        motor_frame.pack(pady=5)
        
        self.active_motor_var = tk.IntVar(value=self.active_motor)
        self.motor_a_btn = tk.Radiobutton(motor_frame, text="Motor A", variable=self.active_motor_var, 
                                          value=0, command=self.switch_motor)
        self.motor_a_btn.pack(side=tk.LEFT, padx=5)
        self.motor_b_btn = tk.Radiobutton(motor_frame, text="Motor B", variable=self.active_motor_var, 
                                          value=1, command=self.switch_motor)
        self.motor_b_btn.pack(side=tk.LEFT, padx=5)
        self.motor_c_btn = tk.Radiobutton(motor_frame, text="Motor C", variable=self.active_motor_var, 
                                          value=2, command=self.switch_motor)
        self.motor_c_btn.pack(side=tk.LEFT, padx=5)
        
        self.active_motor_label = ttk.Label(self.root, text=f"Active: Motor {'ABC'[self.active_motor]}", 
                                            font=("Arial", 10, "bold"))
        self.active_motor_label.pack(pady=5)

        ttk.Button(self.root, text="Stop", command=self.stop).pack(pady=10)
        ttk.Button(self.root, text="Plot Results", command=self.plot_results).pack(pady=5)

        self.update_gui()

    def switch_motor(self):
        """Switch the active motor and reset PID integral."""
        new_motor = self.active_motor_var.get()
        with self.pid_lock:
            self.active_motor = new_motor
            # Reset PID integral when switching motors
            self.integral = 0.0
            self.prev_error = 0.0
        
        # Return to neutral position for all motors
        self.last_angles = self.servo_neutral_angles.copy()
        self.send_servo_angle(0)  # Send neutral position
        motor_names = ['A', 'B', 'C']
        self.active_motor_label.config(text=f"Active: Motor {motor_names[new_motor]}")
        print(f"[MOTOR] Switched to Motor {motor_names[new_motor]}")

    def update_gui(self):
        if self.running:
            with self.pid_lock:
                self.Kp = self.kp_var.get()
                self.Ki = self.ki_var.get()
                self.Kd = self.kd_var.get()
                self.setpoint = self.setpoint_var.get()
                # Update active motor if changed via UI
                if self.active_motor_var.get() != self.active_motor:
                    self.switch_motor()

            self.kp_label.config(text=f"Kp: {self.Kp:.1f}")
            self.ki_label.config(text=f"Ki: {self.Ki:.1f}")
            self.kd_label.config(text=f"Kd: {self.Kd:.1f}")
            self.setpoint_label.config(text=f"Setpoint: {self.setpoint:.3f}")

            try:
                vis_frame = self.display_queue.get_nowait()
                cv2.imshow("Ball Tracking", vis_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    self.running = False
                    self.stop()
            except queue.Empty:
                pass

            self.root.after(50, self.update_gui)

    # --- Plotting ---
    def plot_results(self):
        if not self.time_log:
            print("No data")
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
        ax1.plot(self.time_log, self.position_log, label="Ball Pos")
        ax1.plot(self.time_log, self.setpoint_log, label="Setpoint", linestyle="--")
        ax1.set_ylabel("Position (m)")
        ax1.legend()
        ax1.grid(True)
        ax2.plot(self.time_log, self.control_log, label="Control Output")
        ax2.set_ylabel("Servo Angle (deg)")
        ax2.set_xlabel("Time (s)")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.show()

    def stop(self):
        self.running = False
        cv2.destroyAllWindows()
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
        if self.servo:
            self.servo.close()

    def run(self):
        self.running = True
        Thread(target=self.camera_thread, daemon=True).start()
        Thread(target=self.control_thread, daemon=True).start()
        time.sleep(1)
        self.create_gui()
        self.root.mainloop()
        self.running = False
        print("Controller stopped")

if __name__ == "__main__":
    controller = BasicPIDController()
    controller.run()