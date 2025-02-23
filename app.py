from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import numpy as np
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

# Load the YOLO model
model = YOLO("weights/best.pt")

# Initialize the webcam
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

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
    while True:
        success, frame = camera.read()
        if not success:
            break
        
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
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)