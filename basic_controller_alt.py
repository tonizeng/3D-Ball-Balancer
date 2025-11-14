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
def detect_ball_xy(frame, lower_hsv, upper_hsv):
    """Detect ball in frame and return normalized (x, y) position and visualization frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, None, frame
    largest = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(largest)
    vis_frame = frame.copy()
    if radius > 5:
        cv2.circle(vis_frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
        cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        normalized_x = x / frame.shape[1]
        normalized_y = y / frame.shape[0]
        return True, normalized_x, normalized_y, vis_frame
    return False, None, None, vis_frame

# --- PID Controller ---
class BasicPIDController:
    def __init__(self, config_file="config.json"):
        """Initialize controller using calibration file."""
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        # PID gains
        self.Kp = 0.099
        self.Ki = 0.040
        self.Kd = 0.200

        # Scale factor
        self.scale_factor = self.config['calibration']['pixel_to_meter_ratio'] * self.config['camera']['frame_width']

        # Servo info
        self.servo_port = self.config['servo A']['port']
        self.servo_neutral_angles = [40, 40, 40]
        self.last_angles = self.servo_neutral_angles.copy()
        self.servo = None
        
        # Active motor selection (0=A, 1=B, 2=C)
        self.active_motor = 0  # Default to motor A

        # Ball tracking (now 2D)
        self.setpoint = [0.0, 0.0]  # [setpoint_x, setpoint_y]
        self.integral = [0.0, 0.0, 0.0]
        self.prev_error = [0.0, 0.0, 0.0]

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
            print("Warning: Using frame center as platform center")
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
            self.servo = serial.Serial(self.servo_port, 115200)
            time.sleep(2)
            print("[SERVO] Connected")
            return True
        except Exception as e:
            print(f"[SERVO] Failed: {e}")
            return False

    def send_servo_angles(self, outputs):
        """Send control outputs to all motors."""
        self.last_angles = [int(np.clip(o, 0, 120)) for o in outputs]
        cmd = f"{self.last_angles[0]} {self.last_angles[1]} {self.last_angles[2]}\n"
        print(cmd)
        try:
            if self.servo:
                self.servo.write(cmd.encode())
        except Exception as e:
            print(f"[SERVO] Send failed: {e}")

    # --- PID ---
    def update_pid_all(self, positions, dt=0.033):
        outputs = []
        # Project setpoint onto each motor axis
        setpoint = self.setpoint if isinstance(self.setpoint, list) else [0.0, 0.0]
        for i in range(3):
            # Calculate setpoint projection for this motor
            if self.peg_points_3d and len(self.peg_points_3d) == 3:
                mid_x = self.platform_center[0]
                mid_y = self.platform_center[1]
                peg_point = self.peg_points_3d[i]
                peg_x = peg_point[0]
                peg_y = peg_point[1]
                dir_x = peg_x - mid_x
                dir_y = peg_y - mid_y
                dir_length = np.sqrt(dir_x**2 + dir_y**2)
                if dir_length > 0:
                    dir_x /= dir_length
                    dir_y /= dir_length
                    setpoint_proj = setpoint[0] * dir_x + setpoint[1] * dir_y
                else:
                    setpoint_proj = setpoint[0]
            else:
                setpoint_proj = setpoint[0]
            # Error is position minus setpoint projection
            error = positions[i] - setpoint_proj
            P = self.Kp * error
            self.integral[i] += error * dt
            I = self.Ki * self.integral[i]
            derivative = (error - self.prev_error[i]) / dt
            D = self.Kd * derivative
            self.prev_error[i] = error
            pid_output = P + I + D
            center_angle = 40.0
            output = center_angle + pid_output
            output = np.clip(output, 0, 120)
            outputs.append(output)
        return outputs

    def calculate_motor_positions(self, ball_x_norm, ball_y_norm, center_x, center_y, frame_width, frame_height):
        """Calculate ball position along each motor's axis."""
        ball_x_pixel = ball_x_norm * frame_width
        ball_y_pixel = ball_y_norm * frame_height
        dx = ball_x_pixel - center_x
        dy = ball_y_pixel - center_y
        positions = []
        for i in range(3):
            if self.peg_points_3d and len(self.peg_points_3d) == 3:
                peg_point = self.peg_points_3d[i]
                peg_x = peg_point[0]
                peg_y = peg_point[1]
                dir_x = peg_x - center_x
                dir_y = peg_y - center_y
                dir_length = np.sqrt(dir_x**2 + dir_y**2)
                if dir_length > 0:
                    dir_x /= dir_length
                    dir_y /= dir_length
                    projection = dx * dir_x + dy * dir_y
                else:
                    projection = dx
            else:
                projection = dx
            position_m = projection * self.scale_factor
            positions.append(position_m)
        return positions

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
            
            found, x_norm, y_norm, vis_frame = detect_ball_xy(frame, self.lower_hsv, self.upper_hsv)
            
            # Use platform center from calibration (centroid of 3 peg points)
            # Frame is now exactly the calibration size, so no scaling needed
            mid_x = self.platform_center[0]
            mid_y = self.platform_center[1]
            
            if found:
                positions = self.calculate_motor_positions(
                    x_norm, y_norm, mid_x, mid_y, frame.shape[1], frame.shape[0])
                
                try:
                    if self.position_queue.full():
                        self.position_queue.get_nowait()
                    self.position_queue.put_nowait(positions)
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
            
            # Draw target setpoint marker (always at 2D setpoint position)
            with self.pid_lock:
                setpoint = self.setpoint
            if isinstance(setpoint, list) and len(setpoint) == 2:
                # Convert setpoint (meters) to pixel coordinates relative to platform center
                # The setpoint is in meters, so convert to pixels using scale_factor
                setpoint_x_m, setpoint_y_m = setpoint
                # Platform center in pixels
                center_x, center_y = mid_x, mid_y
                # Setpoint in pixels
                setpoint_x_pix = int(center_x + setpoint_x_m / self.scale_factor)
                setpoint_y_pix = int(center_y + setpoint_y_m / self.scale_factor)
                # Ensure within frame bounds
                setpoint_x_pix = np.clip(setpoint_x_pix, 0, frame.shape[1] - 1)
                setpoint_y_pix = np.clip(setpoint_y_pix, 0, frame.shape[0] - 1)
                # Draw a red cross marker at setpoint position
                marker_size = 15
                cv2.line(vis_frame, 
                         (setpoint_x_pix - marker_size, setpoint_y_pix), 
                         (setpoint_x_pix + marker_size, setpoint_y_pix), 
                         (0, 0, 255), 3)  # Red horizontal line
                cv2.line(vis_frame, 
                         (setpoint_x_pix, setpoint_y_pix - marker_size), 
                         (setpoint_x_pix, setpoint_y_pix + marker_size), 
                         (0, 0, 255), 3)  # Red vertical line
                # Draw a red circle around the setpoint
                cv2.circle(vis_frame, (setpoint_x_pix, setpoint_y_pix), marker_size + 5, (0, 0, 255), 2)
                # Label the setpoint
                cv2.putText(vis_frame, f"Setpoint: x={setpoint_x_m:.3f}, y={setpoint_y_m:.3f}m", 
                            (setpoint_x_pix + marker_size + 5, setpoint_y_pix - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
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
                positions = self.position_queue.get(timeout=0.1)
                # Robust check: skip if not a list of 3 valid numbers
                if (not isinstance(positions, list) or len(positions) != 3 or any(p is None for p in positions)):
                    print(f"[CONTROL] Skipping invalid positions: {positions}")
                    continue
                outputs = self.update_pid_all(positions)
                self.send_servo_angles(outputs)
                current_time = time.time() - self.start_time
                self.time_log.append(current_time)
                self.position_log.append(positions)
                self.setpoint_log.append(self.setpoint.copy())
                self.control_log.append(outputs)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[CONTROL] Error: {e}")
                break

    # --- GUI ---
    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("PID Controller")
        self.root.geometry("520x450")

        ttk.Label(self.root, text="PID Gains", font=("Arial", 18, "bold")).pack(pady=10)

        # Kp
        ttk.Label(self.root, text="Kp").pack()
        self.kp_var = tk.DoubleVar(value=self.Kp)
        kp_slider = ttk.Scale(self.root, from_=0, to=2, variable=self.kp_var, orient=tk.HORIZONTAL, length=500)
        kp_slider.pack()
        self.kp_label = ttk.Label(self.root, text=f"Kp: {self.Kp:.3f}")
        self.kp_label.pack()
        self.kp_entry = ttk.Entry(self.root, width=8)
        self.kp_entry.insert(0, f"{self.Kp:.3f}")
        self.kp_entry.pack()
        self.kp_entry.bind("<Return>", lambda e: self.set_gain_from_entry('kp'))

        # Ki
        ttk.Label(self.root, text="Ki").pack()
        self.ki_var = tk.DoubleVar(value=self.Ki)
        ki_slider = ttk.Scale(self.root, from_=0, to=2, variable=self.ki_var, orient=tk.HORIZONTAL, length=500)
        ki_slider.pack()
        self.ki_label = ttk.Label(self.root, text=f"Ki: {self.Ki:.3f}")
        self.ki_label.pack()
        self.ki_entry = ttk.Entry(self.root, width=8)
        self.ki_entry.insert(0, f"{self.Ki:.3f}")
        self.ki_entry.pack()
        self.ki_entry.bind("<Return>", lambda e: self.set_gain_from_entry('ki'))

        # Kd
        ttk.Label(self.root, text="Kd").pack()
        self.kd_var = tk.DoubleVar(value=self.Kd)
        kd_slider = ttk.Scale(self.root, from_=0, to=2, variable=self.kd_var, orient=tk.HORIZONTAL, length=500)
        kd_slider.pack()
        self.kd_label = ttk.Label(self.root, text=f"Kd: {self.Kd:.3f}")
        self.kd_label.pack()
        self.kd_entry = ttk.Entry(self.root, width=8)
        self.kd_entry.insert(0, f"{self.Kd:.3f}")
        self.kd_entry.pack()
        self.kd_entry.bind("<Return>", lambda e: self.set_gain_from_entry('kd'))

        # Setpoint sliders for X and Y
        ttk.Label(self.root, text="Setpoint X (m)").pack()
        frame_width = self.config['camera']['frame_width']
        frame_height = self.config['camera']['frame_height']
        scale = self.scale_factor
        # Allow setpoint to cover the entire camera feed
        x_range = (frame_width / 2) / scale
        y_range = (frame_height / 2) / scale
        self.setpoint_x_var = tk.DoubleVar(value=self.setpoint[0])
        spx_slider = ttk.Scale(self.root, from_=-x_range, to=x_range, variable=self.setpoint_x_var, orient=tk.HORIZONTAL, length=500)
        spx_slider.pack()
        spx_slider.config(takefocus=1)
        ttk.Label(self.root, text="Setpoint Y (m)").pack()
        self.setpoint_y_var = tk.DoubleVar(value=self.setpoint[1])
        spy_slider = ttk.Scale(self.root, from_=-y_range, to=y_range, variable=self.setpoint_y_var, orient=tk.HORIZONTAL, length=500)
        spy_slider.pack()
        spy_slider.config(takefocus=1)
        self.setpoint_label = ttk.Label(self.root, text=f"Setpoint: x={self.setpoint[0]:.3f}, y={self.setpoint[1]:.3f}")
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

    def set_gain_from_entry(self, gain):
        """Update gain variable and slider from entry field."""
        try:
            if gain == 'kp':
                val = float(self.kp_entry.get())
                self.kp_var.set(val)
            elif gain == 'ki':
                val = float(self.ki_entry.get())
                self.ki_var.set(val)
            elif gain == 'kd':
                val = float(self.kd_entry.get())
                self.kd_var.set(val)
        except Exception:
            pass

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
        self.send_servo_angles(0)  # Send neutral position
        motor_names = ['A', 'B', 'C']
        self.active_motor_label.config(text=f"Active: Motor {motor_names[new_motor]}")
        print(f"[MOTOR] Switched to Motor {motor_names[new_motor]}")

    def update_gui(self):
        if self.running:
            with self.pid_lock:
                self.Kp = round(self.kp_var.get(), 3)
                self.Ki = round(self.ki_var.get(), 3)
                self.Kd = round(self.kd_var.get(), 3)
                self.setpoint = [self.setpoint_x_var.get(), self.setpoint_y_var.get()]
                if self.active_motor_var.get() != self.active_motor:
                    self.switch_motor()
            self.kp_label.config(text=f"Kp: {self.Kp:.3f}")
            self.ki_label.config(text=f"Ki: {self.Ki:.3f}")
            self.kd_label.config(text=f"Kd: {self.Kd:.3f}")
            # Only update entry fields if they do not have focus
            if self.root.focus_get() != self.kp_entry:
                self.kp_entry.delete(0, tk.END)
                self.kp_entry.insert(0, f"{self.Kp:.3f}")
            if self.root.focus_get() != self.ki_entry:
                self.ki_entry.delete(0, tk.END)
                self.ki_entry.insert(0, f"{self.Ki:.3f}")
            if self.root.focus_get() != self.kd_entry:
                self.kd_entry.delete(0, tk.END)
                self.kd_entry.insert(0, f"{self.Kd:.3f}")
            self.setpoint_label.config(text=f"Setpoint: x={self.setpoint[0]:.3f}, y={self.setpoint[1]:.3f}")
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