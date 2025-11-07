# Ball Detection Module for Computer Vision Ball Tracking System
# To convert the code from 1D to 3D ball balancer system, the following changes are needed:
# 
#
#

import cv2
import numpy as np
import json
import os

class BallDetector:
    """Computer vision ball detector using HSV color space filtering."""
    
    def __init__(self, config_file="config.json"):
        """Initialize ball detector with HSV bounds from config file.
        
        Args:
            config_file (str): Path to JSON config file with HSV bounds and calibration
        """
        # Default HSV bounds for orange ball detection
        self.lower_hsv = np.array([5, 150, 150], dtype=np.uint8)  # Orange lower bound
        self.upper_hsv = np.array([20, 255, 255], dtype=np.uint8)  # Orange upper bound
        self.pixel_to_meter_ratio_x = None
        self.pixel_to_meter_ratio_y = None
        self.scale_factor_x = 1.0  # used to be self.scale_factor 
        self.scale_factor_y = 1.0
        self.last_normalized_position = (0.0, 0.0)
        self.last_position_m = (0.0, 0.0)
        self.reference_frame_width = 640
        self.reference_frame_height = 480
        
        # Load configuration from file if it exists
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Extract expected camera reference dimensions if available
                camera_config = config.get('camera', {})
                self.reference_frame_width = camera_config.get('frame_width', self.reference_frame_width)
                self.reference_frame_height = camera_config.get('frame_height', self.reference_frame_height)

                # Extract HSV color bounds from config
                if 'ball_detection' in config:
                    if config['ball_detection']['lower_hsv']:
                        self.lower_hsv = np.array(config['ball_detection']['lower_hsv'], dtype=np.uint8)
                    if config['ball_detection']['upper_hsv']:
                        self.upper_hsv = np.array(config['ball_detection']['upper_hsv'], dtype=np.uint8)
                
                # Extract per-pixel calibration for X/Y conversion to meters
                calibration = config.get('calibration', {})
                ratio_cfg = calibration.get('pixel_to_meter_ratio')
                if ratio_cfg:
                    if isinstance(ratio_cfg, dict):
                        ratio_x = ratio_cfg.get('x')
                        ratio_y = ratio_cfg.get('y', ratio_x)
                    elif isinstance(ratio_cfg, (list, tuple)) and len(ratio_cfg) >= 2:
                        ratio_x, ratio_y = ratio_cfg[0], ratio_cfg[1]
                    else:
                        ratio_x = ratio_y = ratio_cfg

                    if ratio_x is not None:
                        self.pixel_to_meter_ratio_x = float(ratio_x)
                    if ratio_y is not None:
                        self.pixel_to_meter_ratio_y = float(ratio_y) if ratio_y is not None else None

                if self.pixel_to_meter_ratio_y is None:
                    self.pixel_to_meter_ratio_y = self.pixel_to_meter_ratio_x

                # Update informative scale factors for reference frame
                self._update_reference_scale_factors()
                
                print(f"[BALL_DETECT] Loaded HSV bounds: {self.lower_hsv} to {self.upper_hsv}")
                if self.pixel_to_meter_ratio_x is not None:
                    print(f"[BALL_DETECT] Pixel-to-meter ratio X: {self.pixel_to_meter_ratio_x:.6f}")
                if self.pixel_to_meter_ratio_y is not None:
                    print(f"[BALL_DETECT] Pixel-to-meter ratio Y: {self.pixel_to_meter_ratio_y:.6f}")
                print(f"[BALL_DETECT] Reference scale factors (X/Y): {self.scale_factor_x:.6f} / {self.scale_factor_y:.6f} m/normalized_unit")
                
            except Exception as e:
                print(f"[BALL_DETECT] Config load error: {e}, using defaults")
        else:
            print("[BALL_DETECT] No config file found, using default HSV bounds")

    def _update_reference_scale_factors(self):
        """Compute nominal scale factors based on reference frame dimensions."""
        center_x = self.reference_frame_width / 2.0 if self.reference_frame_width else 0.0
        center_y = self.reference_frame_height / 2.0 if self.reference_frame_height else 0.0

        if self.pixel_to_meter_ratio_x is not None and center_x:
            self.scale_factor_x = self.pixel_to_meter_ratio_x * center_x
        else:
            self.scale_factor_x = 1.0

        if self.pixel_to_meter_ratio_y is not None and center_y:
            self.scale_factor_y = self.pixel_to_meter_ratio_y * center_y
        else:
            if center_x and center_y:
                self.scale_factor_y = self.scale_factor_x * (center_y / center_x)
            else:
                self.scale_factor_y = 1.0

    def detect_ball(self, frame):
        """Detect ball in frame and return detection results.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            found (bool): True if ball detected
            center (tuple | None): (x, y) pixel coordinates of ball center
            radius (float | None): Ball radius in pixels
            position_m (tuple): (x, y) position of ball in meters relative to frame center
        """
        # Convert frame from BGR to HSV color space for better color filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create binary mask using HSV color bounds
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Clean up mask using morphological operations
        mask = cv2.erode(mask, None, iterations=2)  # Remove noise
        mask = cv2.dilate(mask, None, iterations=2)  # Fill gaps
        
        # Find all contours in the cleaned mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.last_normalized_position = (0.0, 0.0)
            self.last_position_m = (0.0, 0.0)
            return False, None, None, (0.0, 0.0)
        
        # Select the largest contour (assumed to be the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum enclosing circle around the contour
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        # Filter out detections that are too small or too large
        if radius < 5 or radius > 100:
            self.last_normalized_position = (0.0, 0.0)
            self.last_position_m = (0.0, 0.0)
            return False, None, None, (0.0, 0.0)
        
        height, width = frame.shape[:2]
        center_x = width / 2.0 if width else 0.0
        center_y = height / 2.0 if height else 0.0

        normalized_x = (x - center_x) / center_x if center_x else 0.0
        # Flip Y so positive is forward/upward relative to image center
        normalized_y = (center_y - y) / center_y if center_y else 0.0

        if self.pixel_to_meter_ratio_x is not None and center_x:
            scale_x = self.pixel_to_meter_ratio_x * center_x
        else:
            scale_x = 1.0

        if self.pixel_to_meter_ratio_y is not None and center_y:
            scale_y = self.pixel_to_meter_ratio_y * center_y
        else:
            scale_y = 1.0

        position_x_m = normalized_x * scale_x
        position_y_m = normalized_y * scale_y

        self.scale_factor_x = scale_x  # Maintain legacy attribute for X axis
        self.scale_factor_y = scale_y
        self.last_normalized_position = (normalized_x, normalized_y)
        self.last_position_m = (position_x_m, position_y_m)
        
        return True, (int(x), int(y)), radius, self.last_position_m

    def draw_detection(self, frame, show_info=True):
        """Detect ball and draw detection overlay on frame.
        
        Args:
            frame: Input BGR image frame
            show_info (bool): Whether to display position information text
            
        Returns:
            frame_with_overlay: Frame with detection drawn
            found: True if ball detected
            position_m: (x, y) ball position in meters
        """
        # Perform ball detection
        found, center, radius, position_m = self.detect_ball(frame)
        
        # Create overlay copy for drawing
        overlay = frame.copy()
        
        # Draw vertical and horizontal center reference lines
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2
        cv2.line(overlay, (center_x, 0), (center_x, height), (255, 255, 255), 1)
        cv2.line(overlay, (0, center_y), (width, center_y), (255, 255, 255), 1)
        cv2.putText(overlay, "Center X", (center_x + 5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, "Center Y", (10, center_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if found:
            # Draw circle around detected ball
            cv2.circle(overlay, center, int(radius), (0, 255, 0), 2)  # Green circle
            cv2.circle(overlay, center, 3, (0, 255, 0), -1)  # Green center dot
            
            if show_info:
                # Display ball position information
                pos_x_m, pos_y_m = position_m
                norm_x, norm_y = self.last_normalized_position
                cv2.putText(overlay, f"x_px: {center[0]}", (center[0] - 40, center[1] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(overlay, f"y_px: {center[1]}", (center[0] - 40, center[1] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(overlay, f"x: {pos_x_m:.4f}m ({norm_x:.2f})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(overlay, f"y: {pos_y_m:.4f}m ({norm_y:.2f})", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            if show_info:
                cv2.putText(overlay, "Ball not detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return overlay, found, position_m

# Legacy function for backward compatibility with existing code
def detect_ball_x(frame):
    """Legacy function that matches the old ball_detection.py interface.
    
    This function maintains compatibility with existing code that expects
    the original function signature and return format.
    
    Args:
        frame: Input BGR image frame
        
    Returns:
        found (bool): True if ball detected
        x_normalized (float): Normalized x position (-1 to +1)
        vis_frame (array): Frame with detection overlay
    """
    # Create detector instance using default config
    detector = BallDetector()
    
    # Get detection results with visual overlay
    vis_frame, found, position_m = detector.draw_detection(frame)
    
    if found:
        # Convert back to normalized coordinates for legacy compatibility
        norm_x = detector.last_normalized_position[0]
        x_normalized = np.clip(norm_x, -1.0, 1.0)  # Ensure within bounds
    else:
        x_normalized = 0.0
    
    return found, x_normalized, vis_frame


def detect_ball_xy(frame):
    """Helper for 2D platforms: returns normalized and metric positions.
    
    Args:
        frame: Input BGR image frame
        
    Returns:
        found (bool): True if ball detected
        normalized (tuple): (x, y) normalized positions in range [-1, 1]
        position_m (tuple): (x, y) position in meters
        vis_frame (array): Frame with detection overlay
    """
    detector = BallDetector()
    vis_frame, found, position_m = detector.draw_detection(frame)
    
    if found:
        norm_x, norm_y = detector.last_normalized_position
        normalized = (
            float(np.clip(norm_x, -1.0, 1.0)),
            float(np.clip(norm_y, -1.0, 1.0)),
        )
    else:
        normalized = (0.0, 0.0)
        position_m = (0.0, 0.0)
    
    return found, normalized, position_m, vis_frame

# For testing/calibration when run directly
def main():
    """Test ball detection with current config."""
    detector = BallDetector()
    cap = cv2.VideoCapture(0)  # Use default camera
    
    print("Ball Detection Test")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Resize frame for consistent processing
        frame = cv2.resize(frame, (640, 480))
        
        # Get detection results with overlay
        vis_frame, found, position_m = detector.draw_detection(frame)
        
        # Show detection info in console
        if found:
            pos_x, pos_y = position_m
            norm_x, norm_y = detector.last_normalized_position
            radial_m = (pos_x ** 2 + pos_y ** 2) ** 0.5
            print(
                f"Ball detected -> x: {pos_x:.4f}m ({norm_x:.2f}), "
                f"y: {pos_y:.4f}m ({norm_y:.2f}), r: {radial_m:.4f}m"
            )
        
        # Display frame with detection overlay
        cv2.imshow("Ball Detection Test", vis_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()