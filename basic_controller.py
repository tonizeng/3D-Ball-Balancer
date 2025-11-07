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
from ball_detection import detect_ball_x

class BasicPIDController:
    def __init__(self, config_file="config.json"):
        """Initialize controller, load config, set defaults and queues."""
        # Load experiment and hardware config from JSON file
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        # PID gains (controlled by sliders in GUI)
        self.Kp = 10.0
        self.Ki = 0.0
        self.Kd = 0.0
        # Scale factor for converting from pixels to meters
        self.scale_factor = self.config['calibration']['pixel_to_meter_ratio'] * self.config['camera']['frame_width'] / 2
        # Servo port name and center angle
        self.servo_port = self.config['servo']['port']
        self.neutral_angle = self.config['servo']['neutral_angle']
        self.servo = None
        # Controller-internal state
        self.setpoint = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        # Data logs for plotting results
        self.time_log = []
        self.position_log = []
        self.setpoint_log = []
        self.control_log = []
        self.start_time = None
        # Thread-safe queue for most recent ball position measurement
        self.position_queue = queue.Queue(maxsize=1)
        # Thread-safe queue for display frames (macOS compatibility)
        self.display_queue = queue.Queue(maxsize=1)
        self.running = False    # Main run flag for clean shutdown
        self.camera_ready = False  # Flag to ensure camera is ready before control starts
        self.pid_lock = Lock()  # Thread lock for PID parameters

    def connect_servo(self):
        """Try to open serial connection to servo, return True if success."""
        try:
            self.servo = serial.Serial(self.servo_port, 9600)
            time.sleep(2)
            print("[SERVO] Connected")
            
            # Check if Arduino is responding
            self.servo.flushInput()  # Clear any existing data
            self.servo.write(b'?')  # Send a test command
            time.sleep(0.1)
            
            if self.servo.in_waiting > 0:
                response = self.servo.readline().decode().strip()
                print(f"[SERVO] Arduino response: {response}")
            else:
                print("[SERVO] No response from Arduino - check if code is uploaded")
                
            return True
        except Exception as e:
            print(f"[SERVO] Failed: {e}")
            return False

    def send_servo_angle(self, angle):
        """Send angle command to servo motor (clipped for safety)."""
        if self.servo:
            servo_angle = self.neutral_angle + angle
            servo_angle = int(np.clip(servo_angle, 0, 50))
            try:
                self.servo.write(bytes([servo_angle]))
                print(f"[SERVO] Sent angle: {servo_angle}° (control: {angle:.1f}°)")
            except Exception as e:
                print(f"[SERVO] Send failed: {e}")
        else:
            print(f"[SERVO] No connection - would send angle: {angle:.1f}°")

    def update_pid(self, position, dt=0.033):
        """Perform PID calculation and return control output."""
        # Thread-safe access to PID parameters
        with self.pid_lock:
            kp = self.Kp
            ki = self.Ki
            kd = self.Kd
            setpoint = self.setpoint
        
        error = setpoint - position  # Compute error
        error = error * 100  # Scale error for easier tuning (if needed)
        # Proportional term
        P = kp * error
        # Integral term accumulation
        self.integral += error * dt
        I = ki * self.integral
        # Derivative term calculation
        derivative = (error - self.prev_error) / dt
        D = kd * derivative
        self.prev_error = error
        # PID output (limit to safe beam range)
        output = P + I + D
        output = np.clip(output, -360, 360)
        print(f"[PID] Error: {error:.3f}, P: {P:.1f}, I: {I:.1f}, D: {D:.1f}, Output: {output:.1f}")
        return output

    def camera_thread(self):
        """Dedicated thread for video capture and ball detection."""
        cap = cv2.VideoCapture(self.config['camera']['index'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Wait for camera to initialize and first frame
        frame_count = 0
        while self.running and frame_count < 5:  # Wait for a few frames to ensure camera is stable
            ret, frame = cap.read()
            if ret:
                frame_count += 1
            time.sleep(0.1)
        
        self.camera_ready = True  # Signal that camera is ready
        print("[CAMERA] Camera thread ready, starting main loop")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (320, 240))
            # Detect ball position in frame
            found, x_normalized, vis_frame = detect_ball_x(frame)
            if found:
                # Convert normalized to meters using scale
                position_m = x_normalized * self.scale_factor
                # Always keep latest measurement only
                try:
                    if self.position_queue.full():
                        self.position_queue.get_nowait()
                    self.position_queue.put_nowait(position_m)
                except Exception:
                    pass
            # Send frame to main thread for display (macOS compatibility)
            try:
                if self.display_queue.full():
                    self.display_queue.get_nowait()
                self.display_queue.put_nowait(vis_frame)
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()

    def control_thread(self):
        """Runs PID control loop in parallel with GUI and camera."""
        if not self.connect_servo():
            print("[ERROR] No servo - running in simulation mode")
        
        # Wait for camera to be ready before starting control
        print("[CONTROL] Waiting for camera to be ready...")
        while self.running and not self.camera_ready:
            time.sleep(0.1)
        
        if not self.running:
            return
            
        print("[CONTROL] Camera ready, starting control loop")
        self.start_time = time.time()
        
        while self.running:
            try:
                # Wait for latest ball position from camera
                position = self.position_queue.get(timeout=0.1)
                # Compute control output using PID
                control_output = self.update_pid(position)
                # Send control command to servo (real or simulated)
                self.send_servo_angle(control_output)
                # Log results for plotting
                current_time = time.time() - self.start_time
                self.time_log.append(current_time)
                self.position_log.append(position)
                self.setpoint_log.append(self.setpoint)
                self.control_log.append(control_output)
                print(f"Pos: {position:.3f}m, Output: {control_output:.1f}°")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[CONTROL] Error: {e}")
                break
        if self.servo:
            # Return to neutral on exit
            self.send_servo_angle(0)
            self.servo.close()

    def create_gui(self):
        """Build Tkinter GUI with large sliders and labeled controls."""
        self.root = tk.Tk()
        self.root.title("Basic PID Controller")
        self.root.geometry("520x400")

        # Title label
        ttk.Label(self.root, text="PID Gains", font=("Arial", 18, "bold")).pack(pady=10)

        # Kp slider
        ttk.Label(self.root, text="Kp (Proportional)", font=("Arial", 12)).pack()
        self.kp_var = tk.DoubleVar(value=self.Kp)
        kp_slider = ttk.Scale(self.root, from_=0, to=100, variable=self.kp_var,
                              orient=tk.HORIZONTAL, length=500)
        kp_slider.pack(pady=5)
        self.kp_label = ttk.Label(self.root, text=f"Kp: {self.Kp:.1f}", font=("Arial", 11))
        self.kp_label.pack()

        # Ki slider
        ttk.Label(self.root, text="Ki (Integral)", font=("Arial", 12)).pack()
        self.ki_var = tk.DoubleVar(value=self.Ki)
        ki_slider = ttk.Scale(self.root, from_=0, to=10, variable=self.ki_var,
                              orient=tk.HORIZONTAL, length=500)
        ki_slider.pack(pady=5)
        self.ki_label = ttk.Label(self.root, text=f"Ki: {self.Ki:.1f}", font=("Arial", 11))
        self.ki_label.pack()

        # Kd slider
        ttk.Label(self.root, text="Kd (Derivative)", font=("Arial", 12)).pack()
        self.kd_var = tk.DoubleVar(value=self.Kd)
        kd_slider = ttk.Scale(self.root, from_=0, to=20, variable=self.kd_var,
                              orient=tk.HORIZONTAL, length=500)
        kd_slider.pack(pady=5)
        self.kd_label = ttk.Label(self.root, text=f"Kd: {self.Kd:.1f}", font=("Arial", 11))
        self.kd_label.pack()

        # Setpoint slider
        ttk.Label(self.root, text="Setpoint (meters)", font=("Arial", 12)).pack()
        pos_min = self.config['calibration']['position_min_m']
        pos_max = self.config['calibration']['position_max_m']
        self.setpoint_var = tk.DoubleVar(value=self.setpoint)
        setpoint_slider = ttk.Scale(self.root, from_=pos_min, to=pos_max,
                                   variable=self.setpoint_var,
                                   orient=tk.HORIZONTAL, length=500)
        setpoint_slider.pack(pady=5)
        self.setpoint_label = ttk.Label(self.root, text=f"Setpoint: {self.setpoint:.3f}m", font=("Arial", 11))
        self.setpoint_label.pack()

        # Button group for actions
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)
        ttk.Button(button_frame, text="Reset Integral",
                   command=self.reset_integral).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Plot Results",
                   command=self.plot_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop",
                   command=self.stop).pack(side=tk.LEFT, padx=5)

        # Schedule periodic GUI update
        self.update_gui()

    def update_gui(self):
        """Reflect latest values from sliders into program and update display."""
        if self.running:
            # Thread-safe update of PID parameters
            with self.pid_lock:
                self.Kp = self.kp_var.get()
                self.Ki = self.ki_var.get()
                self.Kd = self.kd_var.get()
                self.setpoint = self.setpoint_var.get()
            
            # Update displayed values
            self.kp_label.config(text=f"Kp: {self.Kp:.1f}")
            self.ki_label.config(text=f"Ki: {self.Ki:.1f}")
            self.kd_label.config(text=f"Kd: {self.Kd:.1f}")
            self.setpoint_label.config(text=f"Setpoint: {self.setpoint:.3f}m")
            
            # Handle OpenCV display in main thread (macOS compatibility)
            self.update_display()
            
            # Call again after 50 ms (if not stopped)
            self.root.after(50, self.update_gui)

    def update_display(self):
        """Handle OpenCV display in main thread (macOS compatibility)."""
        try:
            # Get latest frame from camera thread
            vis_frame = self.display_queue.get_nowait()
            cv2.imshow("Ball Tracking", vis_frame)
            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:  # ESC exits
                self.running = False
                self.stop()
        except queue.Empty:
            pass  # No new frame available
        except Exception as e:
            print(f"[DISPLAY] Error: {e}")

    def reset_integral(self):
        """Clear integral error in PID (button handler)."""
        self.integral = 0.0
        print("[RESET] Integral term reset")

    def plot_results(self):
        """Show matplotlib plots of position and control logs."""
        if not self.time_log:
            print("[PLOT] No data to plot")
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        # Ball position trace
        ax1.plot(self.time_log, self.position_log, label="Ball Position", linewidth=2)
        ax1.plot(self.time_log, self.setpoint_log, label="Setpoint",
                 linestyle="--", linewidth=2)
        ax1.set_ylabel("Position (m)")
        ax1.set_title(f"Basic PID Control (Kp={self.Kp:.1f}, Ki={self.Ki:.1f}, Kd={self.Kd:.1f})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Control output trace
        ax2.plot(self.time_log, self.control_log, label="Control Output",
                 color="orange", linewidth=2)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Beam Angle (degrees)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def stop(self):
        """Stop everything and clean up threads and GUI."""
        self.running = False
        # Try to safely close all windows/resources
        try:
            cv2.destroyAllWindows()
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        """Entry point: starts threads, launches GUI mainloop."""
        print("[INFO] Starting Basic PID Controller")
        print("Use sliders to tune PID gains in real-time")
        print("Close camera window or click Stop to exit")
        self.running = True

        # Start camera thread first
        print("[INFO] Starting camera thread...")
        cam_thread = Thread(target=self.camera_thread, daemon=True)
        cam_thread.start()
        
        # Wait a moment for camera to initialize
        time.sleep(1)
        
        # Start control thread (it will wait for camera to be ready)
        print("[INFO] Starting control thread...")
        ctrl_thread = Thread(target=self.control_thread, daemon=True)
        ctrl_thread.start()
        
        # Wait a moment for control thread to connect to servo
        time.sleep(1)

        # Build and run GUI in main thread
        print("[INFO] Starting GUI...")
        self.create_gui()
        self.root.mainloop()

        # After GUI ends, stop everything
        self.running = False
        print("[INFO] Controller stopped")

if __name__ == "__main__":
    try:
        controller = BasicPIDController()
        controller.run()
    except FileNotFoundError:
        print("[ERROR] config.json not found. Run simple_autocal.py first.")
    except Exception as e:
        print(f"[ERROR] {e}")
