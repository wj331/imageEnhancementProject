import uuid
import requests
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
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], uniqueFileName).replace('\\', '/'))
        
        file_url = f"http://localhost:5000/uploads/{uniqueFileName}"
        print("file saved to url: ", file_url)
        return jsonify({'filePath': file_url}), 200
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

    img_url = data['filePath']
    response = requests.get(img_url, stream= True)
    if response.status_code != 200:
        return jsonify({'error': 'Failed to retrieve image from backend url'}), 400
    
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Failed to decode image'}), 400
    
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
                    x_min = max(0, min(img.shape[1], int(x_min)))
                    y_min = max(0, min(img.shape[0], int(y_min)))
                    x_max = max(0, min(img.shape[1], int(x_max)))
                    y_max = max(0, min(img.shape[0], int(y_max)))

                    label = model.names[int(cls)] if hasattr(model, "names") else f"class_{int(cls)}"

                    detections.append({
                        'x': x_min,
                        'y': y_min,
                        'width': float(x_max - x_min),
                        'height': int(y_max - y_min),
                        'label': label,
                        'confidence': round(float(conf), 2)
                    })#returns json object
    
    return jsonify({'detections': detections})

@app.route('/brighten', methods=['POST'])
def brighten():
    data = request.get_json()
    if not data or 'filePath' not in data:
        return jsonify({'error': 'No image file path provided'}), 400
    
    img_url = data['filePath']
    print("brighten image url", img_url)

    #download image from url
    response = requests.get(img_url, stream= True)
    if response.status_code != 200:
        return jsonify({'error': 'Failed to retrieved image from backend url'}), 400
    
    #convert image to openCV format
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'failed to decode image'}), 400

    #check for brightness value
    hdr = hdr_brightness(img)
    print(f"pre-brighten hdr: {hdr}")

    if 1.5 <= hdr < 4: #sufficient
        return jsonify({'enhanced_image_path': img_url}), 200

    # Enhance the image using CLAHE
    enhanced_img = enhance_image(img)
    
    #create enhanced image path
    file_name = f"enhanced_{os.path.basename(img_url)}"
    enhanced_img_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name).replace('\\', '/')

    #save the enhanced image
    cv2.imwrite(enhanced_img_path, enhanced_img)

    enhanced_img_url = f"http://localhost:5000/uploads/{file_name}"
    return jsonify({'enhanced_image_path': enhanced_img_url}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
