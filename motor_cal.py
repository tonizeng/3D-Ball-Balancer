# Simple Auto Calibration System for Ball and Beam Control
# Interactive calibration tool for color detection and geometry
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



@dataclass
class ServoConfig:
    """
    Maintain different calibration data for each axis/motor on the platform
    """
    servo_port: str = "/dev/cu.usbmodem11101"  # Servo communication port
    neutral_angle: List[float] = field(default_factory=lambda: [30, 0, 0])
    peg_points: List[int] = field(default_factory=list)  # Beam endpoint pixel coords


class SimpleAutoCalibrator:
    """Interactive calibration system for ball and beam control setup."""
    
    def __init__(self):
        """Initialize calibration parameters and default values."""
        # Physical system parameters
        self.BEAM_LENGTH_M = 0.30  # Known beam length in meters
        
        # Camera configuration
        self.CAM_INDEX = 0  # Default camera index
        self.FRAME_W, self.FRAME_H = 800, 600  # Frame dimensions
        
        # Calibration state tracking
        self.current_frame = None  # Current video frame
        self.phase = "color"  # Current phase: "color", "geometry", "complete"
        
        # Color calibration data
        self.hsv_samples = []  # Collected HSV color samples
        self.lower_hsv = None  # Lower HSV bound for ball detection
        self.upper_hsv = None  # Upper HSV bound for ball detection
        
        # Geometry calibration data
        self.pixel_to_meter_ratio = None  # Conversion ratio from pixels to meters
        self.platform_center = None  # Platform center point (centroid of 3 peg points)
        self.peg_points_3d = []  # Three peg points (one per servo) for center calculation
        
        # Three motor config data class instances, one for each motor
        self.servo_A = ServoConfig(neutral_angle=[40, 40, 40])
        self.servo_B = ServoConfig(neutral_angle=[40, 40, 40])
        self.servo_C = ServoConfig(neutral_angle=[40, 40, 40])

        self.servo_serial = serial.Serial(self.servo_A.servo_port, 115200)
        time.sleep(2)
       
        # Track which servo's peg point we're currently collecting
        self.current_servo_index = 0  # 0=A, 1=B, 2=C
        self.servos = [self.servo_A, self.servo_B, self.servo_C]

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events for interactive calibration.
        
        Args:
            event: OpenCV mouse event type
            x, y: Mouse click coordinates
            flags: Additional event flags
            param: User data (unused)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.phase == "color":
                # Color sampling phase - collect HSV samples at click point
                self.sample_color(x, y)
            elif self.phase == "geometry":
                # Geometry phase - collect one peg point per servo (3 points total)
                if len(self.peg_points_3d) < 3:
                    self.peg_points_3d.append((x, y))
                    servo_names = ["A", "B", "C"]
                    print(f"[GEO] Peg point {len(self.peg_points_3d)}/3 selected for Servo {servo_names[len(self.peg_points_3d)-1]}: ({x}, {y})")
                    
                    if len(self.peg_points_3d) == 3:
                        # Calculate platform center as centroid of 3 peg points
                        self.calculate_platform_center()
                        # Geometry calibration complete
                        self.phase = "complete"
                        print("[GEO] All 3 peg points selected. Center calculated. Calibration complete! Press 's' to save.")

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

    def calculate_platform_center(self):
        """Calculate platform center as centroid of 3 peg points."""
        if len(self.peg_points_3d) != 3:
            print("[ERROR] Need exactly 3 peg points to calculate center")
            return
        
        # Calculate centroid (average of 3 points)
        center_x = sum(p[0] for p in self.peg_points_3d) / 3.0
        center_y = sum(p[1] for p in self.peg_points_3d) / 3.0
        self.platform_center = (center_x, center_y)
        
        print(f"[GEO] Platform center calculated: ({center_x:.1f}, {center_y:.1f})")
        
        # Calculate pixel-to-meter ratio using average distance between peg points
        # Use the distance between the first two points as reference
        if len(self.peg_points_3d) >= 2:
            p1, p2 = self.peg_points_3d[0], self.peg_points_3d[1]
            pixel_distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            # Use a reasonable estimate - you may want to adjust this based on your setup
            # For now, we'll estimate based on average spacing
            if pixel_distance > 0:
                # Estimate: if 3 points form a triangle, use average edge length
                dists = []
                for i in range(3):
                    for j in range(i+1, 3):
                        d = math.sqrt((self.peg_points_3d[j][0] - self.peg_points_3d[i][0])**2 + 
                                     (self.peg_points_3d[j][1] - self.peg_points_3d[i][1])**2)
                        dists.append(d)
                avg_distance = sum(dists) / len(dists)
                # Estimate scale: assume the 3 points span roughly the beam length
                # This is an approximation - you may need to calibrate this separately
                self.pixel_to_meter_ratio = self.BEAM_LENGTH_M / avg_distance if avg_distance > 0 else None
                print(f"[GEO] Pixel-to-meter ratio (estimated): {self.pixel_to_meter_ratio:.6f}")
    
    def calculate_geometry(self, servo_obj):
        """Calculate pixel-to-meter conversion ratio from beam endpoint coordinates.
        Legacy method - kept for backward compatibility if needed."""
        if len(servo_obj.peg_points) < 2:
            return
        p1, p2 = servo_obj.peg_points[0], servo_obj.peg_points[1]
        
        # Calculate pixel distance between beam endpoints
        pixel_distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Convert to meters using known beam length
        if pixel_distance > 0:
            self.pixel_to_meter_ratio = self.BEAM_LENGTH_M / pixel_distance
            print(f"[GEO] Pixel-to-meter ratio: {self.pixel_to_meter_ratio:.6f}")

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
                "pixel_to_meter_ratio": float(self.pixel_to_meter_ratio) if self.pixel_to_meter_ratio else None,
                "platform_center": [float(self.platform_center[0]), float(self.platform_center[1])] if self.platform_center else None,
                "peg_points_3d": [[float(p[0]), float(p[1])] for p in self.peg_points_3d] if self.peg_points_3d else None
            },
            "servo A": {
                "port": str(self.servo_A.servo_port),
                "neutral_angle": int(self.servo_A.neutral_angle[0]),
                "peg_points": [list(p) for p in self.servo_A.peg_points] if self.servo_A.peg_points else None
            },
            "servo B": {
                "port": str(self.servo_B.servo_port),
                "neutral_angle": int(self.servo_B.neutral_angle[1]),
                "peg_points": [list(p) for p in self.servo_B.peg_points] if self.servo_B.peg_points else None
            },
            "servo C": {
                "port": str(self.servo_C.servo_port),
                "neutral_angle": int(self.servo_C.neutral_angle[2]),
                "peg_points": [list(p) for p in self.servo_C.peg_points] if self.servo_C.peg_points else None
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
        
        # Phase-specific instruction text (compute dynamically)
        servo_names = ["A", "B", "C"]
        num_selected = len(self.peg_points_3d)
        
        # Build phase text dictionary
        phase_text = {
            "color": "Click on ball to sample colors. Press 'c' when done.",
            "geometry": f"Click peg point for Servo {servo_names[num_selected] if num_selected < 3 else 'Complete'} ({num_selected}/3)",
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
        
        # Show 3D peg points and platform center
        servo_names = ["A", "B", "C"]
        colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]  # Green, Yellow, Cyan
        
        for i, peg in enumerate(self.peg_points_3d):
            cv2.circle(overlay, peg, 10, colors[i], -1)
            cv2.putText(overlay, f"Servo {servo_names[i]}", (peg[0]+15, peg[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        
        # Draw triangle connecting the 3 peg points
        if len(self.peg_points_3d) == 3:
            cv2.line(overlay, self.peg_points_3d[0], self.peg_points_3d[1], (128, 128, 128), 1)
            cv2.line(overlay, self.peg_points_3d[1], self.peg_points_3d[2], (128, 128, 128), 1)
            cv2.line(overlay, self.peg_points_3d[2], self.peg_points_3d[0], (128, 128, 128), 1)
        
        # Draw platform center if calculated
        if self.platform_center:
            center = (int(self.platform_center[0]), int(self.platform_center[1]))
            cv2.circle(overlay, center, 8, (255, 0, 255), -1)  # Magenta center
            cv2.circle(overlay, center, 15, (255, 0, 255), 2)  # Magenta ring
            cv2.putText(overlay, "CENTER", (center[0]+20, center[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
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
        
        return overlay

    def run(self):
        """Main calibration loop with interactive GUI."""

        # Set the motors to their neutral angles
        cmd = f"{40} {40} {40}\n"  # Neutral angle command
        self.servo_serial.write(cmd.encode())

        # Initialize camera capture
        self.cap = cv2.VideoCapture(self.CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        # Setup OpenCV window and mouse callback
        cv2.namedWindow("Auto Calibration")
        cv2.setMouseCallback("Auto Calibration", self.mouse_callback)
        
        # Display instructions
        print("[INFO] Simple Auto Calibration")
        print("Phase 1: Click on ball to sample colors, press 'c' when done")
        print("Phase 2: Click on 3 peg points (one for each servo A, B, C)")
        print("Press 's' to save when complete, 'q' to quit")
        
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
                    print("[INFO] Color calibration complete. Click on 3 peg points (one for each servo A, B, C).")
                    print("[INFO] These points define the platform geometry. The center will be the centroid of these 3 points.")
            elif key == ord('s') and self.phase == "complete":
                # Save configuration and exit
                self.save_config()
                break
        
        # Clean up resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    """Run calibration when script is executed directly."""
    calibrator = SimpleAutoCalibrator()
    calibrator.run()
