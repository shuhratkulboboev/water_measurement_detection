import numpy as np
import cv2
from tensorflow.keras.models import load_model
import argparse
import paho.mqtt.client as mqtt
import time
import json
import os
import sys
import logging
from tensorflow.config import experimental

# Configure environment for CPU-only operation
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
experimental.set_visible_devices([], 'GPU')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('water_meter.log')
    ]
)
logger = logging.getLogger(__name__)

class WaterMeterDetector:
    def __init__(self):
        self.contour_params = {
            'min_area': 2000,
            'max_area': 20000,
            'epsilon_factor': 0.03,
            'aspect_ratio_range': (0.7, 1.3),
            'solidity_threshold': 0.85
        }
        self.frame_count = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 20
        self.last_valid_result = None
        
        self._load_model()
        self._parse_args()
        self._setup_mqtt()
        self._setup_video_capture()

    def _load_model(self):
        """Load the digit detection model with error handling"""
        try:
            self.model = load_model("models/DigitDetector_130epochs.h5")
            self.model.compile(optimizer='adam', 
                             loss='categorical_crossentropy', 
                             metrics=['accuracy'])
            logger.info("Model loaded and compiled successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            sys.exit(1)

    def _parse_args(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description="Water meter digit detection")
        parser.add_argument("--mode", type=int, choices=[0, 1], default=1,
                          help="0: image mode, 1: video mode")
        parser.add_argument("--img_path", type=str, default='images/15.jpg',
                          help="Path to test image (mode 0 only)")
        parser.add_argument("--video_path", type=str, 
                          default='rtsp://vizora.ddns.net:8554/watermeter',
                          help="RTSP stream URL (mode 1 only)")
        parser.add_argument("--debug", action='store_true',
                          help="Enable debug image saving")
        parser.add_argument("--output_dir", type=str, default='debug_output',
                          help="Directory for debug images")
        args = parser.parse_args()
        
        self.mode = args.mode
        self.img_path = args.img_path
        self.video_path = args.video_path
        self.debug = args.debug
        self.output_dir = args.output_dir
        
        if self.debug and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created debug output directory: {self.output_dir}")

    def _setup_video_capture(self):
        """Initialize video capture with enhanced RTSP support"""
        if self.mode == 0:
            return
            
        transport_protocols = ['tcp', 'udp', 'http']
        
        for protocol in transport_protocols:
            url = f"{self.video_path}?{protocol}"
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            if self.cap.isOpened():
                # Configure stream properties
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 10)
                logger.info(f"Connected to video stream using {protocol} protocol")
                return
                
        logger.error("Failed to connect to video stream with all protocols")
        sys.exit(1)

    def _setup_mqtt(self):
        """Initialize MQTT client with reconnect logic"""
        self.mqtt_broker = "vizora.ddns.net"
        self.mqtt_port = 1883
        self.mqtt_topic = "VITMMB09/JDN3HW"
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            logger.info("MQTT connection initiated")
        except Exception as e:
            logger.error(f"Initial MQTT connection failed: {e}")

    def _on_mqtt_connect(self, client, userdata, flags, reason_code, properties):
        """MQTT connection callback"""
        if reason_code == 0:
            logger.info("Connected to MQTT broker")
        else:
            logger.error(f"MQTT connection failed with code {reason_code}")

    def _on_mqtt_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        """MQTT disconnection callback"""
        logger.warning(f"Disconnected from MQTT broker (code: {reason_code})")
        time.sleep(5)
        try:
            client.reconnect()
        except Exception as e:
            logger.error(f"MQTT reconnect failed: {e}")

    def _get_image(self):
        """Get image from source with error handling"""
        if self.mode == 0:
            img = cv2.imread(self.img_path)
            if img is None:
                logger.error(f"Failed to read image: {self.img_path}")
                return None
            return cv2.resize(img, (640, 480))
            
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from video stream")
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                logger.error("Max consecutive failures reached, exiting...")
                sys.exit(1)
            return None
            
        self.consecutive_failures = 0
        return cv2.resize(frame, (640, 480))

    def _preprocess_image(self, img):
        """Enhanced image preprocessing pipeline"""
        if img is None:
            return None
            
        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(enhanced, 255, 
                                      cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY_INV, 21, 10)
        
        # Morphological operations
        kernel = np.ones((3,3), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed = cv2.dilate(processed, kernel, iterations=1)
        
        if self.debug:
            self._debug_save(enhanced, "enhanced")
            self._debug_save(processed, "threshold")
            
        return processed

    def _find_meter_contour(self, processed):
        """Improved contour detection with dynamic parameters"""
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.contour_params['min_area'] < area < self.contour_params['max_area']):
                continue
                
            # Calculate contour solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area if hull_area > 0 else 0
            
            # Approximate contour polygon
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.contour_params['epsilon_factor'] * peri, True)
            
            # Check for quadrilateral shape and solidity
            if len(approx) == 4 and solidity > self.contour_params['solidity_threshold']:
                x,y,w,h = cv2.boundingRect(approx)
                aspect_ratio = float(w)/h
                if (self.contour_params['aspect_ratio_range'][0] < aspect_ratio < 
                    self.contour_params['aspect_ratio_range'][1]):
                    valid_contours.append((cnt, area, solidity))
        
        # Return best contour by area and solidity
        if valid_contours:
            return max(valid_contours, key=lambda x: (x[1], x[2]))[0]
        return None

    def _four_point_transform(self, image, pts):
        """Perform perspective transform on detected display"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        if self.debug:
            self._debug_save(warped, "warped")
            
        return warped

    def _extract_digits(self, warped):
        """Extract and preprocess digits from the display"""
        # Resize to standard display size
        display = cv2.resize(warped, (160, 32))
        
        digits = []
        for i in range(5):  # For 5 digit display
            x_start = 3 + i * 32
            x_end = x_start + 32
            digit = display[2:30, x_start:x_end]
            
            # Preprocess digit
            digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
            digit = cv2.equalizeHist(digit)
            digit = cv2.resize(digit, (28, 28))
            digit = digit / 255.0
            digit = digit.reshape(1, 28, 28, 1)
            
            digits.append(digit)
            
            if self.debug:
                self._debug_save((digit * 255).astype(np.uint8), f"digit_{i}")
                
        return digits

    def _recognize_digits(self, digits):
        """Recognize digits using the trained model"""
        results = []
        confidences = []
        
        for digit in digits:
            pred = self.model.predict(digit, verbose=0)
            digit_class = np.argmax(pred)
            confidence = np.max(pred)
            
            if confidence > 0.7:  # Confidence threshold
                results.append(int(digit_class))
                confidences.append(float(confidence))
            else:
                results.append(-1)  # Unknown digit
                confidences.append(0.0)
                
        return results, confidences

    def _publish_results(self, digits, confidences):
        """Publish results to MQTT"""
        payload = {
            "timestamp": int(time.time()),
            "digits": digits,
            "confidences": confidences,
            "frame_count": self.frame_count
        }
        
        try:
            result = self.mqtt_client.publish(self.mqtt_topic, json.dumps(payload))
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Published to MQTT: {payload}")
            else:
                logger.warning(f"MQTT publish failed with code {result.rc}")
        except Exception as e:
            logger.error(f"MQTT publish error: {e}")

    def _debug_save(self, img, prefix=""):
        """Save images for debugging"""
        if self.debug:
            timestamp = int(time.time())
            filename = f"{self.output_dir}/{prefix}_{timestamp}_{self.frame_count}.png"
            cv2.imwrite(filename, img)
            logger.debug(f"Saved debug image: {filename}")

    def process_frame(self):
        """Complete frame processing pipeline"""
        # 1. Capture frame
        frame = self._get_image()
        if frame is None:
            return None
            
        # 2. Preprocess
        processed = self._preprocess_image(frame)
        if processed is None:
            return None
            
        # 3. Find display contour
        contour = self._find_meter_contour(processed)
        if contour is None:
            logger.warning("No valid display contour found")
            if self.debug:
                contour_img = frame.copy()
                cv2.drawContours(contour_img, [contour], -1, (0,255,0), 2)
                self._debug_save(contour_img, "contour_fail")
            return None
            
        # 4. Perspective transform
        warped = self._four_point_transform(frame, contour.reshape(4,2))
        
        # 5. Extract and recognize digits
        digits = self._extract_digits(warped)
        digit_values, confidences = self._recognize_digits(digits)
        
        return digit_values, confidences

    def run(self):
        """Main processing loop"""
        last_publish_time = 0
        publish_interval = 5  # seconds
        
        try:
            while True:
                start_time = time.time()
                
                # Process frame
                results = self.process_frame()
                
                if results:
                    digit_values, confidences = results
                    logger.info(f"Detected digits: {digit_values} (confidences: {confidences})")
                    
                    # Publish results periodically
                    current_time = time.time()
                    if current_time - last_publish_time > publish_interval:
                        self._publish_results(digit_values, confidences)
                        last_publish_time = current_time
                    
                    self.last_valid_result = results
                else:
                    # Use last valid result if available
                    if self.last_valid_result and (time.time() - last_publish_time) > publish_interval:
                        digit_values, confidences = self.last_valid_result
                        self._publish_results(digit_values, confidences)
                        last_publish_time = time.time()
                
                self.frame_count += 1
                
                # Control processing rate
                elapsed = time.time() - start_time
                if elapsed < 0.1:  # ~10 FPS max
                    time.sleep(0.1 - elapsed)
                    
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            logger.info("Video capture released")
        
        if hasattr(self, 'mqtt_client'):
            self.mqtt_client.disconnect()
            self.mqtt_client.loop_stop()
            logger.info("MQTT client disconnected")
        
        logger.info("System shutdown complete")

if __name__ == "__main__":
    detector = WaterMeterDetector()
    detector.run()