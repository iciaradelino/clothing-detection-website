from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import numpy as np
from flask_cors import CORS
import logging
from flask import request, jsonify
import base64
import io
from PIL import Image
import os
import requests
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make sure to correctly specify the static folder path
app = Flask(__name__, 
            static_folder='static',
            static_url_path='/static')
CORS(app)

# Load the YOLO model
model = None  # Initialize as None

def download_model():
    """Download the model if it doesn't exist"""
    model_url = os.environ.get('MODEL_URL', '')  # URL to download model from
    model_path = '/opt/render/project/src/weights/best.pt'  # Fixed path in Render's persistent storage
    
    # Create weights directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if not os.path.exists(model_path) and model_url:
        logger.info(f"Downloading model from {model_url}")
        try:
            response = requests.get(model_url)
            response.raise_for_status()
            with open(model_path, 'wb') as f:
                f.write(response.content)
            logger.info("Model downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            return None
    return model_path

def load_model():
    global model
    if model is None:
        # Check for local model first
        local_model_path = "weights/best.pt"
        
        if os.path.exists(local_model_path):
            logger.info(f"Loading local model from {local_model_path}")
            model = YOLO(local_model_path)
        else:
            # Try to get model path from environment variable or use default path
            model_path = download_model()
            
            try:
                if model_path and os.path.exists(model_path):
                    logger.info(f"Loading model from {model_path}")
                    model = YOLO(model_path)
                else:
                    logger.warning("Model file not found, using default YOLOv8n model")
                    model = YOLO("yolov8n.pt")  # Fallback to a default model
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                # Fall back to the coco model which should work
                logger.info("Falling back to COCO model")
                model = YOLO("yolov8n.pt", task='detect')
            
    return model

# Improve camera handling with more robust error handling
camera = None
camera_lock = None

def get_camera():
    global camera
    try:
        if camera is None:
            logger.info("Initializing webcam...")
            camera = cv2.VideoCapture(0)
            
            # Check if camera opened successfully and is returning frames
            if not camera.isOpened():
                logger.error("Failed to open webcam")
                return None
                
            # Read a test frame to ensure camera is working
            success, _ = camera.read()
            if not success:
                logger.error("Webcam opened but not returning frames")
                cleanup_camera()
                return None
                
            logger.info("Webcam initialized successfully")
        return camera
    except Exception as e:
        logger.error(f"Error initializing webcam: {str(e)}")
        cleanup_camera()
        return None

def cleanup_camera():
    global camera
    if camera is not None:
        try:
            logger.info("Releasing webcam resources")
            camera.release()
        except Exception as e:
            logger.error(f"Error releasing webcam: {str(e)}")
        finally:
            camera = None
            logger.info("Webcam resources released")

# Add this after the model initialization
CATEGORY_COLORS = {
    'upper_body': (255, 0, 0),    # Red for upper body items
    'lower_body': (0, 255, 0),    # Green for lower body items
    'footwear': (0, 0, 255),      # Blue for shoes
    'accessories': (255, 165, 0),  # Orange for accessories
    'outerwear': (128, 0, 128),   # Purple for outerwear
    'other': (255, 255, 0)        # Yellow for other items
}

def get_category_color(class_name):
    # Define category mappings
    categories = {
        'upper_body': ['t-shirt', 'shirt', 'blouse', 'tank top', 'sweater'],
        'lower_body': ['pants', 'jeans', 'shorts', 'skirt'],
        'footwear': ['shoes', 'boots', 'sneakers', 'sandals'],
        'accessories': ['hat', 'cap', 'scarf', 'tie', 'belt', 'bag'],
        'outerwear': ['jacket', 'coat', 'hoodie']
    }
    
    # Find category for class
    for category, items in categories.items():
        if any(item in class_name.lower() for item in items):
            return CATEGORY_COLORS[category]
    return CATEGORY_COLORS['other']

# Modify the generate_frames function
# Add this list after your CATEGORY_COLORS definition
EXCLUDED_CLASSES = ['sleeve', 'neckline']  # Add any classes you want to ignore

def generate_frames():
    """Generate video frames with object detection"""
    global camera
    
    # Get the camera and model
    camera = get_camera()
    model = load_model()
    
    if camera is None or model is None:
        cleanup_camera()
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n'
               b'Error initializing webcam or model\r\n')
        return
        
    try:
        # Set a timeout to avoid hanging if frames aren't received
        frame_timeout_count = 0
        max_timeout_count = 10
        
        while True:
            success, frame = camera.read()
            
            if not success:
                frame_timeout_count += 1
                logger.warning(f"Failed to read frame from webcam ({frame_timeout_count}/{max_timeout_count})")
                
                # If we've failed too many times in a row, exit the loop
                if frame_timeout_count >= max_timeout_count:
                    logger.error("Too many failed frames, stopping video feed")
                    break
                
                # Add a short delay before trying again
                import time
                time.sleep(0.1)
                continue
            
            # Reset timeout counter if we got a frame
            frame_timeout_count = 0
            
            # Resize frame for better performance
            frame = cv2.resize(frame, (640, 480))
            
            # Skip processing every other frame for performance
            if hasattr(generate_frames, 'skip_frame'):
                generate_frames.skip_frame = not generate_frames.skip_frame
                if generate_frames.skip_frame:
                    # Just show the raw frame without detections
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                         b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    continue
            else:
                generate_frames.skip_frame = False
            
            try:
                results = model.predict(frame, conf=0.5)
                
                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        class_name = model.names[cls_id]
                        
                        # Skip drawing if class is in excluded list
                        if class_name.lower() in EXCLUDED_CLASSES:
                            continue
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = box.conf[0].item()
                        label = f"{class_name} {confidence:.2f}"
                        color = get_category_color(class_name)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                logger.error(f"Error in model prediction: {str(e)}")
                # Add error text to the frame
                cv2.putText(frame, "Model error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Optimize JPEG encoding
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if not ret:
                logger.error("Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except GeneratorExit:
        # This exception occurs when the client disconnects
        logger.info("Client disconnected from video feed")
    except Exception as e:
        logger.error(f"Error in generate_frames: {str(e)}")
    finally:
        cleanup_camera()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    try:
        camera = get_camera()
        if camera is None:
            return "Error: Could not initialize webcam", 500
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error in video_feed route: {str(e)}")
        cleanup_camera()
        return str(e), 500

# Add this new route
@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    # Get the image from the request
    file = request.files['image']
    img = Image.open(file.stream)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Load and run detection
    model = load_model()
    results = model.predict(img_cv, conf=0.5)
    detected_items = []
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]
            
            # Skip excluded classes
            if class_name.lower() in EXCLUDED_CLASSES:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            label = f"{class_name} {confidence:.2f}"
            
            # Crop the detected item
            item_img = img_cv[y1:y2, x1:x2]
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', item_img)
            img_str = base64.b64encode(buffer).decode()
            
            detected_items.append({
                'image': img_str,
                'label': label
            })
    
    return jsonify({'items': detected_items})

# Add stop_webcam route to properly release camera resources
@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    try:
        cleanup_camera()
        return jsonify({"status": "success", "message": "Webcam stopped successfully"})
    except Exception as e:
        logger.error(f"Error stopping webcam: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Load the model when the app starts
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)