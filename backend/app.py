import uuid
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_socketio import SocketIO
import numpy as np
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO
from .services.image_service import enhance_image, hdr_brightness
from .utils.image_utils import numpy_to_base64
import os
import cv2

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"UPLOAD_FOLDER path: {os.path.abspath(UPLOAD_FOLDER)}")

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']

    if file and file.filename != '':
        ext = os.path.splitext(file.filename)[1]
        uniqueFileName = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uniqueFileName).replace('\\', '/')
        file.save(file_path)
        return jsonify({'filePath': file_path}), 200
    else:
        return jsonify({'error': 'No image file provided'}), 400
    
@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if not data or 'filePath' not in data:
        return jsonify({'error': 'No image filepath provided'}), 400

    file_path = data['filePath']
    if not os.path.exists(file_path):
        return jsonify({'error': 'Invalid image filepath provided'}), 400
    img = Image.open(file_path)

    # model = YOLO(r"C:/Users/wenji/OneDrive/Desktop/Y3S2/ATAP/Image Enhancement Project/image-enhancement-app/backend/uno.pt")
    model = YOLO(r"C:/Users/wenji/OneDrive/Desktop/Y3S2/ATAP/Image Enhancement Project/image-enhancement-app/backend/yolov8s.pt")
    # Perform inference
    results = model(img)

    # Extract bounding boxes and labels
    detections = []
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                box_data = box.data.cpu().numpy()
                for row in box_data:
                    x_min, y_min, x_max, y_max, conf, cls = row
                    x_min = max(0, min(img.width, int(x_min)))
                    y_min = max(0, min(img.height, int(y_min)))
                    x_max = max(0, min(img.width, int(x_max)))
                    y_max = max(0, min(img.height, int(y_max)))

                    detections.append({
                        'x': x_min,
                        'y': y_min,
                        'width': float(x_max - x_min),
                        'height': int(y_max - y_min),
                        'label': result.names[int(cls)],
                        'confidence': round(float(conf), 2)
                    })#returns json object
    
    return jsonify({'detections': detections})

@app.route('/brighten', methods=['POST'])
def brighten():
    data = request.get_json()
    if not data or 'filePath' not in data:
        return jsonify({'error': 'No image file path provided'}), 400
    
    file_path = data['filePath']
    if not os.path.exists(file_path):
        return jsonify({'error': 'Invalid image file path provided'}), 400
    print("brighten image path", file_path)
    fileName = file_path.split('/')[-1]

    #check for brightness value
    hdr = hdr_brightness(file_path)
    print(f"pre-brighten hdr: {hdr}")

    if 1.5 <= hdr < 4: #sufficient
        return jsonify({'enhanced_image_path': file_path})
        
    # Enhance the image using CLAHE
    enhanced_img = enhance_image(file_path)
    
    #create enhanced image path
    file_name = f"enhanced_{fileName}"
    enhanced_img_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name).replace('\\', '/')

    #save the enhanced image
    cv2.imwrite(enhanced_img_path, enhanced_img)

    return jsonify({'enhanced_image_path': enhanced_img_path}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
