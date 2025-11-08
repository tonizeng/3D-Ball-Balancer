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

# --------------------------
# Simple 2D ball detection
# --------------------------
def detect_ball_xy(frame):
    """Detect ball in frame and return normalized (x, y) and visualization."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Hard-coded HSV bounds for a yellow/orange ball
    lower = np.array([8, 202, 168])
    upper = np.array([19, 235, 200])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)

    # Find contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 2:
            # Normalize coordinates (0=center)
            x_norm = (x - frame.shape[1]/2)/(frame.shape[1]/2)
            y_norm = (y - frame.shape[0]/2)/(frame.shape[0]/2)
            vis_frame = frame.copy()
            cv2.circle(vis_frame, (int(x), int(y)), int(radius), (0,255,0), 2)
            return True, x_norm, y_norm, vis_frame
    return False, 0.0, 0.0, frame

# --------------------------
# Stewart Platform Controller
# --------------------------
class StewartPIDController:
    def __init__(self, config_file="config.json"):
        # Load config
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        # PID gains
        self.Kp = 10.0
        self.Ki = 0.0
        self.Kd = 0.0

        # Scale factor (pixels -> meters)
        self.scale_factor = self.config['calibration']['pixel_to_meter_ratio']

        # Servo info
        self.servo_port = self.config['servo']['port']
        self.servo_neutral_angles = self.config['servo']['neutral_angles']  # list of 3
        self.last_angles = self.servo_neutral_angles.copy()
        self.servo = None

        # Controller state
        self.setpoint = [0.0, 0.0]   # X and Y
        self.integral = [0.0, 0.0]
        self.prev_error = [0.0, 0.0]

        # Logs
        self.time_log = []
        self.pos_log = []
        self.set_log = []
        self.ctrl_log = []

        # Queues
        self.position_queue = queue.Queue(maxsize=1)
        self.display_queue = queue.Queue(maxsize=1)

        self.running = False
        self.camera_ready = False
        self.pid_lock = Lock()

    # --------------------------
    # Servo communication
    # --------------------------
    def connect_servo(self):
        try:
            self.servo = serial.Serial(self.servo_port, 9600)
            time.sleep(2)
            print("[SERVO] Connected")
            return True
        except Exception as e:
            print(f"[SERVO] Failed: {e}")
            return False

    def send_servo_angles(self, angles):
        """Send array of 3 angles; None = keep last value."""
        for i in range(3):
            if angles[i] is not None:
                self.last_angles[i] = int(np.clip(angles[i] + self.servo_neutral_angles[i], 0, 120))
        cmd = f"{self.last_angles[0]} {self.last_angles[1]} {self.last_angles[2]}\n"
        if self.servo:
            try:
                self.servo.write(cmd.encode())
                print(f"[SERVO] Sent: {cmd.strip()}")
            except Exception as e:
                print(f"[SERVO] Send failed: {e}")

    # --------------------------
    # PID calculation (2D)
    # --------------------------
    def update_pid(self, pos, dt=0.033):
        """pos = [x, y] in meters"""
        with self.pid_lock:
            kp, ki, kd = self.Kp, self.Ki, self.Kd
            setpoint = self.setpoint.copy()

        error = [setpoint[0]-pos[0], setpoint[1]-pos[1]]
        P = [kp*error[0], kp*error[1]]
        self.integral[0] += error[0]*dt
        self.integral[1] += error[1]*dt
        I = [ki*self.integral[0], ki*self.integral[1]]
        derivative = [(error[0]-self.prev_error[0])/dt, (error[1]-self.prev_error[1])/dt]
        D = [kd*derivative[0], kd*derivative[1]]
        self.prev_error = error.copy()
        output = [P[0]+I[0]+D[0], P[1]+I[1]+D[1]]  # X, Y control
        print(f"[PID] Error: {error}, Output: {output}")
        return output

    # --------------------------
    # Camera
    # --------------------------
    def find_camera_index(self):
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                return i
        return 0

    def camera_thread(self):
        index = self.find_camera_index()
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Wait for a few frames
        frame_count = 0
        while self.running and frame_count < 5:
            ret, frame = cap.read()
            if ret: frame_count += 1
            time.sleep(0.1)

        self.camera_ready = True
        print("[CAMERA] Ready")

        while self.running:
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.resize(frame, (320,240))
            found, x_norm, y_norm, vis_frame = detect_ball_xy(frame)
            if found:
                try:
                    if self.position_queue.full(): self.position_queue.get_nowait()
                    self.position_queue.put_nowait([x_norm*self.scale_factor, y_norm*self.scale_factor])
                except:
                    pass
            try:
                if self.display_queue.full(): self.display_queue.get_nowait()
                self.display_queue.put_nowait(vis_frame)
            except: pass
        cap.release()
        cv2.destroyAllWindows()

    # --------------------------
    # Control loop
    # --------------------------
    def control_thread(self):
        if not self.connect_servo():
            print("[CONTROL] Running without real servo")
        while self.running and not self.camera_ready:
            time.sleep(0.1)
        print("[CONTROL] Starting loop")
        self.start_time = time.time()

        while self.running:
            try:
                pos = self.position_queue.get(timeout=0.1)
                ctrl = self.update_pid(pos)
                # Map X,Y control to 3 Stewart motors (simple example)
                # You will need actual kinematics for real platform
                angles = [
                    ctrl[0] + ctrl[1],  # Motor A
                    ctrl[0] - ctrl[1],  # Motor B
                    -ctrl[0]          # Motor C
                ]
                self.send_servo_angles(angles)
                t = time.time() - self.start_time
                self.time_log.append(t)
                self.pos_log.append(pos.copy())
                self.set_log.append(self.setpoint.copy())
                self.ctrl_log.append(angles.copy())
            except queue.Empty:
                continue

    # --------------------------
    # GUI
    # --------------------------
    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Stewart PID Controller")
        self.root.geometry("520x400")

        # PID sliders
        self.kp_var = tk.DoubleVar(value=self.Kp)
        self.ki_var = tk.DoubleVar(value=self.Ki)
        self.kd_var = tk.DoubleVar(value=self.Kd)
        self.set_x_var = tk.DoubleVar(value=self.setpoint[0])
        self.set_y_var = tk.DoubleVar(value=self.setpoint[1])

        ttk.Label(self.root, text="Kp").pack()
        ttk.Scale(self.root, from_=0, to=100, variable=self.kp_var, orient=tk.HORIZONTAL).pack()
        ttk.Label(self.root, text="Ki").pack()
        ttk.Scale(self.root, from_=0, to=10, variable=self.ki_var, orient=tk.HORIZONTAL).pack()
        ttk.Label(self.root, text="Kd").pack()
        ttk.Scale(self.root, from_=0, to=20, variable=self.kd_var, orient=tk.HORIZONTAL).pack()
        ttk.Label(self.root, text="Setpoint X").pack()
        ttk.Scale(self.root, from_=-0.15, to=0.15, variable=self.set_x_var, orient=tk.HORIZONTAL).pack()
        ttk.Label(self.root, text="Setpoint Y").pack()
        ttk.Scale(self.root, from_=-0.15, to=0.15, variable=self.set_y_var, orient=tk.HORIZONTAL).pack()

        # Buttons
        ttk.Button(self.root, text="Stop", command=self.stop).pack(pady=5)
        ttk.Button(self.root, text="Plot", command=self.plot_results).pack(pady=5)

        self.update_gui()
        self.root.mainloop()

    def update_gui(self):
        if self.running:
            with self.pid_lock:
                self.Kp = self.kp_var.get()
                self.Ki = self.ki_var.get()
                self.Kd = self.kd_var.get()
                self.setpoint = [self.set_x_var.get(), self.set_y_var.get()]
            # Display latest frame
            try:
                frame = self.display_queue.get_nowait()
                cv2.imshow("Ball", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    self.stop()
            except: pass
            self.root.after(50, self.update_gui)

    # --------------------------
    # Plot
    # --------------------------
    def plot_results(self):
        if not self.time_log: return
        fig, axs = plt.subplots(2,1, figsize=(10,6))
        pos_array = np.array(self.pos_log)
        set_array = np.array(self.set_log)
        axs[0].plot(self.time_log, pos_array[:,0], label="X")
        axs[0].plot(self.time_log, pos_array[:,1], label="Y")
        axs[0].plot(self.time_log, set_array[:,0], "--", label="X_set")
        axs[0].plot(self.time_log, set_array[:,1], "--", label="Y_set")
        axs[0].legend()
        axs[0].set_ylabel("Position (m)")
        ctrl_array = np.array(self.ctrl_log)
        axs[1].plot(self.time_log, ctrl_array[:,0], label="Motor A")
        axs[1].plot(self.time_log, ctrl_array[:,1], label="Motor B")
        axs[1].plot(self.time_log, ctrl_array[:,2], label="Motor C")
        axs[1].legend()
        axs[1].set_ylabel("Angles (deg)")
        axs[1].set_xlabel("Time (s)")
        plt.show()

    # --------------------------
    # Stop
    # --------------------------
    def stop(self):
        self.running = False
        cv2.destroyAllWindows()
        try: self.root.quit(); self.root.destroy()
        except: pass
        if self.servo: self.servo.close()

    # --------------------------
    # Run
    # --------------------------
    def run(self):
        self.running = True
        Thread(target=self.camera_thread, daemon=True).start()
        Thread(target=self.control_thread, daemon=True).start()
        self.create_gui()


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    try:
        controller = StewartPIDController()
        controller.run()
    except FileNotFoundError:
        print("[ERROR] config.json not found.")
    except Exception as e:
        print(f"[ERROR] {e}")
