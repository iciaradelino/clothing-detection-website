from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import numpy as np
from flask_cors import CORS
import logging
from flask import request, jsonify
import base64
import io
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)

# Load the YOLO model
model = YOLO("weights/best.pt")

# Initialize the webcam
camera = None

def get_camera():
    global camera
    try:
        if camera is None:
            logger.info("Initializing webcam...")
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                logger.error("Failed to open webcam")
                return None
            logger.info("Webcam initialized successfully")
        return camera
    except Exception as e:
        logger.error(f"Error initializing webcam: {str(e)}")
        return None

def cleanup_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

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
    camera = get_camera()
    if camera is None:
        return
        
    try:
        while True:
            success, frame = camera.read()
            if not success:
                logger.error("Failed to read frame from webcam")
                break
            
            # Resize frame for better performance
            frame = cv2.resize(frame, (640, 480))
            
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
            
            # Optimize JPEG encoding
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        logger.error(f"Error in generate_frames: {str(e)}")
    finally:
        cleanup_camera()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    camera = get_camera()
    if camera is None:
        return "Error: Could not initialize webcam", 500
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Add this new route
@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    # Get the image from the request
    file = request.files['image']
    img = Image.open(file.stream)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Run detection
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

if __name__ == '__main__':
    app.run(debug=True)