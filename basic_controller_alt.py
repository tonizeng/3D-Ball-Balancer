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
        self.Kp = 10.0
        self.Ki = 0.0
        self.Kd = 0.0

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

        # Perpendicular line (use servo A endpoints)
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
        """Send single control output as servo A angle; keep B,C neutral."""
        self.last_angles[0] = int(np.clip(output + self.servo_neutral_angles[0], 0, 120))
        self.last_angles[1] = self.servo_neutral_angles[1]
        self.last_angles[2] = self.servo_neutral_angles[2]
        cmd = f"{self.last_angles[0]} {self.last_angles[1]} {self.last_angles[2]}\n"
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

        error = setpoint - position
        error *= 100  # scale
        P = kp * error
        self.integral += error * dt
        I = ki * self.integral
        derivative = (error - self.prev_error) / dt
        D = kd * derivative
        self.prev_error = error
        output = np.clip(P + I + D, -360, 360)
        return output

    # --- Camera Thread ---
    def camera_thread(self):
        cap = cv2.VideoCapture(self.config['camera']['index'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (320, 240))
            found, x_norm, vis_frame = detect_ball_x(frame, self.lower_hsv, self.upper_hsv)
            if found:
                # Compute perpendicular line midpoint
                x1, y1 = self.lineA[0]
                x2, y2 = self.lineA[1]
                mid_x = (x1 + x2)/2
                pixel_offset = x_norm*frame.shape[1] - mid_x
                position_m = pixel_offset * self.scale_factor
                try:
                    if self.position_queue.full():
                        self.position_queue.get_nowait()
                    self.position_queue.put_nowait(position_m)
                except Exception:
                    pass
            # Draw perpendicular line
            cv2.line(vis_frame, self.lineA[0], self.lineA[1], (255, 0, 0), 2)
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
        pos_min = self.config['servo A']['position_min_m'] or -0.15
        pos_max = self.config['servo A']['position_max_m'] or 0.15
        self.setpoint_var = tk.DoubleVar(value=self.setpoint)
        sp_slider = ttk.Scale(self.root, from_=pos_min, to=pos_max, variable=self.setpoint_var, orient=tk.HORIZONTAL, length=500)
        sp_slider.pack()
        self.setpoint_label = ttk.Label(self.root, text=f"Setpoint: {self.setpoint:.3f}")
        self.setpoint_label.pack()

        ttk.Button(self.root, text="Stop", command=self.stop).pack(pady=10)
        ttk.Button(self.root, text="Plot Results", command=self.plot_results).pack(pady=5)

        self.update_gui()

    def update_gui(self):
        if self.running:
            with self.pid_lock:
                self.Kp = self.kp_var.get()
                self.Ki = self.ki_var.get()
                self.Kd = self.kd_var.get()
                self.setpoint = self.setpoint_var.get()

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