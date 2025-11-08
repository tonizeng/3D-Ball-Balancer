
# simple_autocal.py
# Simple Auto Calibration System for Ball and Beam Control (multi-servo)
import cv2
import numpy as np
import json
import math
import serial
import time
from datetime import datetime

class SimpleAutoCalibrator:
    """Interactive calibration system for ball and beam control setup."""

    def __init__(self):
        """Initialize calibration parameters and default values."""
        # Physical system parameters
        self.BEAM_LENGTH_M = 0.2  # Known beam length in meters

        # Camera configuration
        self.CAM_INDEX = 0  # Default camera index
        self.FRAME_W, self.FRAME_H = 640, 480  # Frame dimensions

        # Calibration state tracking
        self.current_frame = None  # Current video frame
        self.phase = "color"  # Current phase: "color", "geometry", "limits", "complete"

        # Color calibration data
        self.hsv_samples = []  # Collected HSV color samples
        self.lower_hsv = None  # Lower HSV bound for ball detection
        self.upper_hsv = None  # Upper HSV bound for ball detection

        # Geometry calibration data
        self.peg_points = []  # Beam endpoint pixel coordinates
        self.pixel_to_meter_ratio = None  # Conversion ratio from pixels to meters

        # Servo hardware configuration (single Arduino USB controls all servos)
        self.servo = None  # Serial connection to Arduino controlling servos
        self.servo_port = "/dev/cu.usbmodem11101"  # Update to your port
        # Neutral angle PER servo (3 servos)
        self.neutral_angles = [30, 30, 30]

        # Position limit results
        self.position_min = None  # Minimum ball position in meters
        self.position_max = None  # Maximum ball position in meters

    def connect_servo(self):
        """Establish serial connection to Arduino (single port) for servo control."""
        try:
            self.servo = serial.Serial(self.servo_port, 9600, timeout=1)
            time.sleep(2)  # Allow time for connection to stabilize
            print("[SERVO] Connected to Arduino on", self.servo_port)
            return True
        except Exception as e:
            print("[SERVO] Failed to connect - limits will be estimated:", e)
            self.servo = None
            return False

    def send_servo_angles(self, angles):
        """Send three servo angles over serial as 'a1 a2 a3\\n'."""
        if not self.servo:
            return
        # Clip each angle to a safe range (example 0..120)
        clipped = [int(np.clip(a, 0, 120)) for a in angles]
        cmd = f"{clipped[0]} {clipped[1]} {clipped[2]}\n"
        try:
            self.servo.write(cmd.encode("utf-8"))
            # small pause to allow Arduino to process
            time.sleep(0.05)
            print("[SERVO] ->", cmd.strip())
        except Exception as e:
            print("[SERVO] Send failed:", e)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events for interactive calibration."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.phase == "color":
                # Color sampling phase - collect HSV samples at click point
                self.sample_color(x, y)
            elif self.phase == "geometry" and len(self.peg_points) < 2:
                # Geometry phase - collect beam endpoint coordinates
                self.peg_points.append((x, y))
                print(f"[GEO] Peg {len(self.peg_points)} selected")
                if len(self.peg_points) == 2:
                    self.calculate_geometry()

    def sample_color(self, x, y):
        """Sample HSV color values in a 5x5 region around click point."""
        if self.current_frame is None:
            return

        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                px, py = x + dx, y + dy
                if 0 <= px < hsv.shape[1] and 0 <= py < hsv.shape[0]:
                    self.hsv_samples.append(hsv[py, px])

        if self.hsv_samples:
            samples = np.array(self.hsv_samples)
            h_margin = max(5, (np.max(samples[:, 0]) - np.min(samples[:, 0])) * 0.1)
            s_margin = max(10, (np.max(samples[:, 1]) - np.min(samples[:, 1])) * 0.15)
            v_margin = max(10, (np.max(samples[:, 2]) - np.min(samples[:, 2])) * 0.15)

            self.lower_hsv = [
                max(0, np.min(samples[:, 0]) - h_margin),
                max(0, np.min(samples[:, 1]) - s_margin),
                max(0, np.min(samples[:, 2]) - v_margin)
            ]
            self.upper_hsv = [
                min(179, np.max(samples[:, 0]) + h_margin),
                min(255, np.max(samples[:, 1]) + s_margin),
                min(255, np.max(samples[:, 2]) + v_margin)
            ]
            print(f"[COLOR] Samples: {len(self.hsv_samples)}")
            print(f"[COLOR] lower: {self.lower_hsv}, upper: {self.upper_hsv}")

    def calculate_geometry(self):
        """Calculate pixel-to-meter conversion ratio from beam endpoint coordinates."""
        p1, p2 = self.peg_points
        # Correct Euclidean distance (squared terms)
        pixel_distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        if pixel_distance <= 0:
            print("[GEO] Invalid peg points - pixel distance <= 0")
            return
        self.pixel_to_meter_ratio = self.BEAM_LENGTH_M / pixel_distance
        print(f"[GEO] Pixel-to-meter ratio: {self.pixel_to_meter_ratio:.6f}")
        self.phase = "limits"

    def detect_ball_position(self, frame):
        """Detect ball in frame and return position in meters from center."""
        if not self.lower_hsv or not self.pixel_to_meter_ratio:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(self.lower_hsv, dtype=np.uint8)
        upper = np.array(self.upper_hsv, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest)
        if radius < 5:
            return None

        center_x = frame.shape[1] // 2
        pixel_offset = x - center_x
        meters_offset = pixel_offset * self.pixel_to_meter_ratio
        return meters_offset

    def find_limits_automatically(self):
        """Use servo motor(s) to automatically find ball position limits."""
        if not self.servo:
            # Estimate limits without servo if connection failed
            self.position_min = -self.BEAM_LENGTH_M / 2
            self.position_max = self.BEAM_LENGTH_M / 2
            print("[LIMITS] Estimated without servo")
            return

        print("[LIMITS] Finding limits with servo (sending 3-angle sets)...")
        positions = []

        # Define test angle sets for three servos (customize for your hardware)
        test_angle_sets = [
            [self.neutral_angles[0], self.neutral_angles[1], self.neutral_angles[2]],
            [self.neutral_angles[0] + 20, self.neutral_angles[1] - 10, self.neutral_angles[2] + 5],
            [self.neutral_angles[0] - 20, self.neutral_angles[1] + 10, self.neutral_angles[2] - 5]
        ]
        print("[DEBUG] Test angle sets:", test_angle_sets)

        for i, angles in enumerate(test_angle_sets):
            print(f"[DEBUG] Starting test {i+1}/{len(test_angle_sets)}: angles {angles}")
            self.send_servo_angles(angles)
            time.sleep(2)  # Wait for ball to settle

            angle_positions = []
            start_time = time.time()
            while time.time() - start_time < 1.0:
                ret, frame = self.cap.read()
                if ret:
                    pos = self.detect_ball_position(frame)
                    if pos is not None:
                        angle_positions.append(pos)
                time.sleep(0.03)

            if angle_positions:
                avg_pos = float(np.mean(angle_positions))
                positions.append(avg_pos)
                print(f"[LIMITS] Angles {angles}: {avg_pos:.4f}m ({len(angle_positions)} measurements)")
            else:
                print(f"[LIMITS] Angles {angles}: No ball detected!")

        # Return servos to neutral
        self.send_servo_angles(self.neutral_angles)

        if len(positions) >= 2:
            self.position_min = float(min(positions))
            self.position_max = float(max(positions))
            print(f"[LIMITS] Range: {self.position_min:.4f}m to {self.position_max:.4f}m")
        else:
            print("[LIMITS] Failed to find limits reliably; using estimates")
            self.position_min = -self.BEAM_LENGTH_M / 2
            self.position_max = self.BEAM_LENGTH_M / 2

    def save_config(self):
        """Save all calibration results to config.json file."""
        config = {
            "timestamp": datetime.now().isoformat(),
            "beam_length_m": float(self.BEAM_LENGTH_M),
            "camera": {
                "index": int(self.CAM_INDEX),
                "frame_width": int(self.FRAME_W),
                "frame_height": int(self.FRAME_H)
            },
            "ball_detection": {
                "lower_hsv": [float(x) for x in self.lower_hsv] if self.lower_hsv is not None else None,
                "upper_hsv": [float(x) for x in self.upper_hsv] if self.upper_hsv is not None else None
            },
            "calibration": {
                "pixel_to_meter_ratio": float(self.pixel_to_meter_ratio) if self.pixel_to_meter_ratio else None,
                "position_min_m": float(self.position_min) if self.position_min else None,
                "position_max_m": float(self.position_max) if self.position_max else None
            },
            "servo": {
                "port": str(self.servo_port),
                "neutral_angles": [int(a) for a in self.neutral_angles]
            }
        }

        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("[SAVE] Configuration saved to config.json")

    def draw_overlay(self, frame):
        """Draw calibration status and instructions overlay on frame."""
        overlay = frame.copy()

        phase_text = {
            "color": "Click on ball to sample colors. Press 'c' when done.",
            "geometry": "Click on beam endpoints (2 points).",
            "limits": "Press 'l' to find limits automatically.",
            "complete": "Calibration complete! Press 's' to save."
        }
        cv2.putText(overlay, f"Phase: {self.phase}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, phase_text.get(self.phase, ""), (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if self.hsv_samples:
            cv2.putText(overlay, f"Color samples: {len(self.hsv_samples)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for i, peg in enumerate(self.peg_points):
            cv2.circle(overlay, peg, 8, (0, 255, 0), -1)
            cv2.putText(overlay, f"Peg {i+1}", (peg[0]+10, peg[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if len(self.peg_points) == 2:
            cv2.line(overlay, self.peg_points[0], self.peg_points[1], (255, 0, 0), 2)

        if self.lower_hsv:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array(self.lower_hsv, dtype=np.uint8)
            upper = np.array(self.upper_hsv, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(largest)
                if radius > 5:
                    cv2.circle(overlay, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(overlay, (int(x), int(y)), 3, (0, 255, 255), -1)
                    if self.pixel_to_meter_ratio:
                        pos = self.detect_ball_position(frame)
                        if pos is not None:
                            cv2.putText(overlay, f"Pos: {pos:.4f}m",
                                       (int(x)+20, int(y)+20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if self.position_min is not None and self.position_max is not None:
            cv2.putText(overlay, f"Limits: {self.position_min:.4f}m to {self.position_max:.4f}m",
                       (10, overlay.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return overlay

    def run(self):
        """Main calibration loop with interactive GUI."""
        self.cap = cv2.VideoCapture(self.CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        cv2.namedWindow("Auto Calibration")
        cv2.setMouseCallback("Auto Calibration", self.mouse_callback)

        self.connect_servo()

        print("[INFO] Simple Auto Calibration")
        print("Phase 1: Click on ball to sample colors, press 'c' when done")
        print("Phase 2: Click on beam endpoints")
        print("Phase 3: Press 'l' to find limits")
        print("Press 's' to save, 'q' to quit")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.current_frame = frame
            display = self.draw_overlay(frame)
            cv2.imshow("Auto Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c') and self.phase == "color":
                if self.hsv_samples:
                    self.phase = "geometry"
                    print("[INFO] Color calibration complete. Click on beam endpoints.")
            elif key == ord('l') and self.phase == "limits":
                self.find_limits_automatically()
                self.phase = "complete"
            elif key == ord('s') and self.phase == "complete":
                self.save_config()
                break

        self.cap.release()
        cv2.destroyAllWindows()
        if self.servo:
            try:
                self.send_servo_angles(self.neutral_angles)
                self.servo.close()
            except Exception:
                pass

if __name__ == "__main__":
    calibrator = SimpleAutoCalibrator()
    calibrator.run()
