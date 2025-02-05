from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
from flask_cors import CORS
import torch
from PIL import Image
import io
from ultralytics import YOLO
import base64
from LIME.LIME import LIME

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image file provided'}), 400

    # Load image
    base64_str = data['image'].split(',')[1]
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes))

    model = YOLO('yolov8s.pt')
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
                    detections.append({
                        'x': int(x_min),
                        'y': int(y_min),
                        'width': float(x_max - x_min),
                        'height': int(y_max - y_min),
                        'label': result.names[int(cls)],
                        'confidence': float(conf)
                    })#returns json object

    return jsonify({'detections': detections})

@app.route('/brighten', methods=['POST'])
def brighten():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image file provided'}), 400
    
    # Enhance the image using LIME
    enhanced_img = enhance_image(data['image'])
    
    # Convert the enhanced image to base64
    enhanced_base64 = numpy_to_base64(enhanced_img)
    
    # Return the enhanced image in the response
    return jsonify({'enhanced_image_url': f"data:image/png;base64,{enhanced_base64}"})

def enhance_image(image):
    """
    Enhance the input image using the LIME algorithm.
    """
    # Initialize the LIME model
    img_data = base64.b64decode(image.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert the image to LAB color space
    lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel_clahe = clahe.apply(l_channel)

    # Merge the enhanced L channel with the original A and B channels
    lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))

    # Convert back to BGR color space (numpy)
    enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return enhanced_img

def numpy_to_base64(img):
    """
    Convert a NumPy array (OpenCV image) to a base64-encoded string.
    """
    # Encode the image into a byte buffer (PNG format)
    _, buffer = cv2.imencode('.png', img)

    # Convert the byte buffer to a base64 string
    base64_str = base64.b64encode(buffer).decode('utf-8')

    return base64_str

def pil_to_base64(img):
    """
    Convert a PIL image to a base64-encoded string.
    """
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_byte_arr = buffered.getvalue()
    base64_str = base64.b64encode(img_byte_arr).decode('utf-8')
    return base64_str


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
