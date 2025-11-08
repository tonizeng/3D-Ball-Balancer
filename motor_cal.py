# Simple Auto Calibration System for Ball and Beam Control
# Interactive calibration tool for color detection, geometry, and servo limits
# Generates config.json file for use with ball tracking controller

import cv2
import numpy as np
import json
import math
import serial
import time
from datetime import datetime
from dataclasses import dataclass, field

from typing import List, Optional

# Need to define this globally to change data we are saving dynamically in the mouse callback.

CURR_MOTOR = None
TURN = 0


@dataclass
class ServoConfig:
    """
    Maintain different calibration data for each axis/motor on the platform
    """
    servo: Optional[object] = None  # Serial connection to servo
    servo_port: str = "/dev/cu.usbmodem11101"  # Servo communication port
    neutral_angle: List[float] = field(default_factory=lambda: [30, 0, 0])
    position_min: Optional[float] = None  # Minimum ball position in meters
    position_max: Optional[float] = None  # Maximum ball position in meters
    peg_points: List[int] = field(default_factory=list)  # Beam endpoint pixel coords


class SimpleAutoCalibrator:
    """Interactive calibration system for ball and beam control setup."""
    
    def __init__(self):
        """Initialize calibration parameters and default values."""
        # Physical system parameters
        self.BEAM_LENGTH_M = 0.30  # Known beam length in meters
        
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
        self.pixel_to_meter_ratio = None  # Conversion ratio from pixels to meters
        
        # Three motor config data class instances, one for each motor
        self.servo_A = ServoConfig(neutral_angle=[60, 0, 0])
        self.servo_B = ServoConfig(neutral_angle=[0, 80, 0])
        self.servo_C = ServoConfig(neutral_angle=[0, 0, 20]) # TODO: doenst fully go to neutral angle, investigate please!


    def connect_servo(self):
        """Establish serial connection to servo motor for automated limit finding.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.servo_A.servo = serial.Serial(self.servo_A.servo_port, 9600)
            time.sleep(2)  # Allow time for connection to stabilize
            print("[SERVO A] Connected")
            
            self.servo_B.servo = serial.Serial(self.servo_B.servo_port, 9600)
            time.sleep(2)  # Allow time for connection to stabilize
            print("[SERVO B] Connected")

            self.servo_C.servo = serial.Serial(self.servo_C.servo_port, 9600)
            time.sleep(2)  # Allow time for connection to stabilize
            print("[SERVO C] Connected")
            return True
        except:
            print("[SERVO] Failed to connect - limits will be estimated")
            return False

    def send_servo_angle(self, angles, servo_obj):
        """Send three servo angles over serial as 'a1 a2 a3\\n'."""
        if not servo_obj:
            return
        # Clip each angle to a safe range (example 0..120)
        clipped = [int(np.clip(a, 0, 120)) for a in angles]
        cmd = f"{clipped[0]} {clipped[1]} {clipped[2]}\n"
        try:
            servo_obj.servo.write(cmd.encode("utf-8"))
            # small pause to allow Arduino to process
            time.sleep(0.05)
            print("[SERVO] ->", cmd.strip())
        except Exception as e:
            print("[SERVO] Send failed:", e)


    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events for interactive calibration.
        
        Args:
            event: OpenCV mouse event type
            x, y: Mouse click coordinates
            flags: Additional event flags
            param: User data (unused)
        """
        if not CURR_MOTOR:
            return
        
        peg_points = CURR_MOTOR.peg_points

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.phase == "color":
                # Color sampling phase - collect HSV samples at click point
                self.sample_color(x, y)
            elif self.phase == "geometry" and len(peg_points) < 2:
                # Geometry phase - collect beam endpoint coordinates
                peg_points.append((x, y))
                print(f"[GEO] Peg {len(peg_points)} selected")
                if len(peg_points) == 2:
                    self.calculate_geometry(CURR_MOTOR)

    def sample_color(self, x, y):
        """Sample HSV color values in a 5x5 region around click point.
        
        Args:
            x, y: Center coordinates for color sampling
        """
        if self.current_frame is None:
            return
        
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        
        # Sample 5x5 region around click point
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                px, py = x + dx, y + dy
                # Check bounds and collect valid samples
                if 0 <= px < hsv.shape[1] and 0 <= py < hsv.shape[0]:
                    self.hsv_samples.append(hsv[py, px])
        
        # Update HSV bounds based on collected samples
        if self.hsv_samples:
            samples = np.array(self.hsv_samples)
            
            # Calculate adaptive margins for each HSV channel
            h_margin = max(5, (np.max(samples[:, 0]) - np.min(samples[:, 0])) * 0.1)
            s_margin = max(10, (np.max(samples[:, 1]) - np.min(samples[:, 1])) * 0.15)
            v_margin = max(10, (np.max(samples[:, 2]) - np.min(samples[:, 2])) * 0.15)
            
            # Set lower bounds with margin
            self.lower_hsv = [
                max(0, np.min(samples[:, 0]) - h_margin),
                max(0, np.min(samples[:, 1]) - s_margin),
                max(0, np.min(samples[:, 2]) - v_margin)
            ]
            
            # Set upper bounds with margin
            self.upper_hsv = [
                min(179, np.max(samples[:, 0]) + h_margin),
                min(255, np.max(samples[:, 1]) + s_margin),
                min(255, np.max(samples[:, 2]) + v_margin)
            ]
            
            print(f"[COLOR] Samples: {len(self.hsv_samples)}")

    def calculate_geometry(self, servo_obj):
        """Calculate pixel-to-meter conversion ratio from beam endpoint coordinates."""
        p1, p2 = servo_obj.peg_points
        
        # Calculate pixel distance between beam endpoints
        pixel_distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Convert to meters using known beam length
        self.pixel_to_meter_ratio = self.BEAM_LENGTH_M / pixel_distance
        print(f"[GEO] Pixel-to-meter ratio: {self.pixel_to_meter_ratio:.6f}")
        
        # Advance to limits calibration phase
        self.phase = "limits"

    def detect_ball_position(self, frame):
        """Detect ball in frame and return position in meters from center.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            float or None: Ball position in meters from center, None if not detected
        """
        if not self.lower_hsv:
            return None
        
        # Convert to HSV and create color mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(self.lower_hsv, dtype=np.uint8)
        upper = np.array(self.upper_hsv, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Clean up mask with morphological operations
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Get largest contour (assumed to be ball)
        largest = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest)
        
        # Filter out very small detections
        if radius < 5:
            return None
        
        # Convert pixel position to meters from center
        center_x = frame.shape[1] // 2
        pixel_offset = x - center_x
        meters_offset = pixel_offset * self.pixel_to_meter_ratio
        
        return meters_offset

    def find_limits_automatically(self, servo_obj):
        """Use servo motor to automatically find ball position limits."""
        if not servo_obj:
            # Estimate limits without servo if connection failed
            global_position_min = -self.BEAM_LENGTH_M / 2
            global_position_max = self.BEAM_LENGTH_M / 2
            print("[LIMITS] Estimated without servo")
            return
        
        print("[LIMITS] Finding limits with servo...")
        positions = []
        
        # Test servo at different angles to find position range
        #test_angles = [self.neutral_angle - 30, self.neutral_angle, self.neutral_angle + 20]
        test_angles = [50, 30, 0]

        print(f"[DEBUG] Test angles: {test_angles}")
        
        for i, angle in enumerate(test_angles):
            
            # Move servo to test angle. We construct an array of angles for the given motor i.e. place zeroes in the correct index to move the right motor.
            angle_array_A = [angle,0,0]
            angle_array_B = [0, angle,0]
            angle_array_C = [0,0,angle]
            global TURN
            angle_array = angle_array_A
            if TURN == 1:
                angle_array = angle_array_B
            elif TURN == 2:
                angle_array = angle_array_C

            print(f"[DEBUG] Starting test {i+1}/3: angle {angle} degrees (turn {TURN})")

            self.send_servo_angle(angle_array, servo_obj)
            time.sleep(2)  # Wait for ball to settle
            
            # Collect multiple position measurements
            angle_positions = []
            start_time = time.time()
            while time.time() - start_time < 1.0:
                ret, frame = self.cap.read()
                if ret:
                    pos = self.detect_ball_position(frame)
                    if pos is not None:
                        angle_positions.append(pos)
                time.sleep(0.05)
            
            # Calculate average position for this angle
            if angle_positions:
                avg_pos = np.mean(angle_positions)
                positions.append(avg_pos)
                print(f"[LIMITS] Angle {angle}: {avg_pos:.4f}m ({len(angle_positions)} measurements)")
            else:
                print(f"[LIMITS] Angle {angle}: No ball detected!")
            print(f"[DEBUG] Completed test {i+1}/3 for angle {angle}")

        TURN += 1

        print(f"[DEBUG] All tests completed. Collected {len(positions)} position measurements")

        # Return servo to neutral position
        self.send_servo_angle(servo_obj.neutral_angle, servo_obj)
        
        # Determine position limits from collected data
        if len(positions) >= 2:
            servo_obj.position_min = min(positions)
            servo_obj.position_max = max(positions)
            print(f"[LIMITS] Range: {servo_obj.position_min:.4f}m to {servo_obj.position_max:.4f}m")
        else:
            print("[LIMITS] Failed to find limits")

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
                "lower_hsv": [float(x) for x in self.lower_hsv] if self.lower_hsv else None,
                "upper_hsv": [float(x) for x in self.upper_hsv] if self.upper_hsv else None
            },
            "calibration": {
                "pixel_to_meter_ratio": float(self.pixel_to_meter_ratio) if self.pixel_to_meter_ratio else None
            },
            "servo A": {
                "port": str(self.servo_A.servo_port),
                "neutral_angle": int(self.servo_A.neutral_angle[0]),
                "position_min_m": float(self.servo_A.position_min) if self.servo_A.position_min else None,
                "position_max_m": float(self.servo_A.position_max) if self.servo_A.position_max else None
            },
            "servo B": {
                "port": str(self.servo_B.servo_port),
                "neutral_angle": int(self.servo_B.neutral_angle[1]),
                "position_min_m": float(self.servo_B.position_min) if self.servo_B.position_min else None,
                "position_max_m": float(self.servo_B.position_max) if self.servo_B.position_max else None
            },
            "servo C": {
                "port": str(self.servo_C.servo_port),
                "neutral_angle": int(self.servo_C.neutral_angle[2]),
                "position_min_m": float(self.servo_C.position_min) if self.servo_C.position_min else None,
                "position_max_m": float(self.servo_C.position_max) if self.servo_C.position_max else None
            }
        }
        
        # Write configuration to JSON file
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("[SAVE] Configuration saved to config.json")

    def draw_overlay(self, frame):
        """Draw calibration status and instructions overlay on frame.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            numpy.ndarray: Frame with overlay graphics and text
        """
        overlay = frame.copy()
        
        # Phase-specific instruction text
        phase_text = {
            "color": "Click on ball to sample colors. Press 'c' when done.",
            "geometry": "Click on beam endpoints (2 points)",
            "limits": "Press 'l' to find limits automatically",
            "complete": "Calibration complete! Press 's' to save"
        }
        
        # Draw current phase and instructions
        cv2.putText(overlay, f"Phase: {self.phase}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, phase_text[self.phase], (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show color calibration progress
        if self.hsv_samples:
            cv2.putText(overlay, f"Color samples: {len(self.hsv_samples)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show geometry calibration points for each motor axis
        
        for i, peg in enumerate(self.servo_A.peg_points):
            cv2.circle(overlay, peg, 8, (0, 255, 0), -1)
            cv2.putText(overlay, f"Peg {i+1}", (peg[0]+10, peg[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
         # Draw line between beam endpoints if both are selected
        if len(self.servo_A.peg_points) == 2:
            cv2.line(overlay, self.servo_A.peg_points[0], self.servo_A.peg_points[1], (255, 0, 0), 2)
            
        for i, peg in enumerate(self.servo_B.peg_points):
            cv2.circle(overlay, peg, 8, (0, 255, 0), -1)
            cv2.putText(overlay, f"Peg {i+1}", (peg[0]+10, peg[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # Draw line between beam endpoints if both are selected
        if len(self.servo_B.peg_points) == 2:
            cv2.line(overlay, self.servo_B.peg_points[0], self.servo_B.peg_points[1], (255, 0, 0), 2)
        
        for i, peg in enumerate(self.servo_C.peg_points):
            cv2.circle(overlay, peg, 8, (0, 255, 0), -1)
            cv2.putText(overlay, f"Peg {i+1}", (peg[0]+10, peg[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # Draw line between beam endpoints if both are selected
        if len(self.servo_C.peg_points) == 2:
            cv2.line(overlay, self.servo_C.peg_points[0], self.servo_C.peg_points[1], (255, 0, 0), 2)
        
        # Show real-time ball detection if color calibration is complete
        if self.lower_hsv:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array(self.lower_hsv, dtype=np.uint8)
            upper = np.array(self.upper_hsv, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Clean up mask
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            # Find and draw detected ball
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(largest)
                if radius > 5:
                    # Draw detection circle
                    cv2.circle(overlay, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(overlay, (int(x), int(y)), 3, (0, 255, 255), -1)
                    
                    # Show position if geometry calibration is complete
                    if self.pixel_to_meter_ratio:
                        pos = self.detect_ball_position(frame)
                        if pos is not None:
                            cv2.putText(overlay, f"Pos: {pos:.4f}m",
                                       (int(x)+20, int(y)+20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Show final results if limit calibration is complete
        if self.servo_A.position_min is not None and self.servo_A.position_max is not None:
            cv2.putText(overlay, f"Limits: {self.servo_A.position_min:.4f}m to {self.servo_A.position_max:.4f}m",
                       (10, overlay.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if self.servo_B.position_min is not None and self.servo_B.position_max is not None:
            cv2.putText(overlay, f"Limits: {self.servo_A.position_min:.4f}m to {self.servo_B.position_max:.4f}m",
                       (10, overlay.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
        if self.servo_C.position_min is not None and self.servo_C.position_max is not None:
            cv2.putText(overlay, f"Limits: {self.servo_A.position_min:.4f}m to {self.servo_C.position_max:.4f}m",
                       (10, overlay.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return overlay

    def run(self):
        """Main calibration loop with interactive GUI."""
        # Initialize camera capture
        self.cap = cv2.VideoCapture(self.CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        # Setup OpenCV window and mouse callback
        cv2.namedWindow("Auto Calibration")
        cv2.setMouseCallback("Auto Calibration", self.mouse_callback)
        
        # Attempt servo connection
        self.connect_servo()
        
        # Display instructions
        print("[INFO] Simple Auto Calibration")
        print("Phase 1: Click on ball to sample colors, press 'c' when done")
        print("Phase 2: Click on beam endpoints")
        print("Phase 3: Press 'l' to find limits")
        print("Press 's' to save, 'q' to quit")
        
        # Main calibration loop
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            self.current_frame = frame
            
            # Draw overlay and display frame
            display = self.draw_overlay(frame)
            cv2.imshow("Auto Calibration", display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit calibration
                break
            elif key == ord('c') and self.phase == "color":
                # Complete color calibration phase
                if self.hsv_samples:
                    self.phase = "geometry"
                    print("[INFO] Color calibration complete. Click on beam endpoints.")
            elif key == ord('l') and self.phase == "limits":
                # Start automatic limit finding for each motor
                CURR_MOTOR = calibrator.servo_A
                self.find_limits_automatically(self.servo_A)
                CURR_MOTOR = calibrator.servo_B
                self.find_limits_automatically(self.servo_B)
                CURR_MOTOR = calibrator.servo_C
                self.find_limits_automatically(self.servo_C)

                self.phase = "complete"
            elif key == ord('s') and self.phase == "complete":
                # Save configuration and exit
                self.save_config()
                break
        
        # Clean up resources
        self.cap.release()
        cv2.destroyAllWindows()
        if self.servo_A:
            self.servo_A.servo.close()
        if self.servo_B:
            self.servo_B.servo.close()
        if self.servo_C:
            self.servo_C.servo.close()

if __name__ == "__main__":
    """Run calibration when script is executed directly."""
    calibrator = SimpleAutoCalibrator()
    CURR_MOTOR = calibrator.servo_A
    calibrator.run()
