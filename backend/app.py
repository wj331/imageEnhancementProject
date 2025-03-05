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
from .services.zerodceEnhancement import enhance_image_zerodce

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
        
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], uniqueFileName).replace('\\', '/')

        image = Image.open(file)
        image = image.resize((400, 300))
        image.save(save_path)
        
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
    
    model = YOLO(r"C:/Users/wenji/OneDrive/Desktop/Y3S2/ATAP/Image Enhancement Project/image-enhancement-app/backend/uno.pt")
    # model = YOLO(r"C:/Users/wenji/OneDrive/Desktop/Y3S2/ATAP/Image Enhancement Project/image-enhancement-app/backend/yolov8s.pt")
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

    if 2.5 <= hdr < 4: #sufficient
        return jsonify({'enhanced_image_path': img_url}), 200

    # # Enhance the image using CLAHE
    # enhanced_img = enhance_image(img)

    #Enhance the image using Zero-DCE
    enhanced_img = enhance_image_zerodce(img)
    
    #create enhanced image path
    file_name = f"enhanced_{os.path.basename(img_url)}"
    enhanced_img_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name).replace('\\', '/')

    #save the enhanced image
    cv2.imwrite(enhanced_img_path, enhanced_img)

    enhanced_img_url = f"http://localhost:5000/uploads/{file_name}"
    return jsonify({'enhanced_image_path': enhanced_img_url}), 200

class Detection:
    def __init__(self, x, y, width, height, label, confidence):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.confidence = confidence

    def get_center(self) -> tuple:
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    def get_area(self) -> float:
        return self.width * self.height
    
@app.route('/calculate-improvement', methods=['POST'])
def calculate_improvement(spatial_threshold = 0.7, size_threshold = 0.7):
    data = request.get_json()
    print('data received!', data)

    if not data or 'uploadedDetections' not in data or 'brightenedDetections' not in data:
        return jsonify({'error': 'Missing required data'}), 400
    
    uploaded_detections = data['uploadedDetections']
    brightened_detections = data['brightenedDetections']

    def convert_to_detection(detection):
        return [Detection(**d) for d in detection]
    
    def calculate_spatial_similarity(detection1, detection2):
        center1 = detection1.get_center()
        center2 = detection2.get_center()
        center_dist = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        
        #normalize by average detection size
        avg_size = (np.sqrt(detection1.get_area()) + np.sqrt(detection2.get_area())) / 2
        normalized_dist = center_dist / avg_size

        #calculate size similarity
        area_ratio = min(detection1.get_area(), detection2.get_area()) / max(detection1.get_area(), detection2.get_area())
        
        #combine distance and size metrics
        return (1 - normalized_dist) * 0.7 + area_ratio * 0.3
    
    orig_dets = convert_to_detection(uploaded_detections)
    bright_dets = convert_to_detection(brightened_detections)

    results = {
        'new_detections': [],
        'lost_detections': [],
        'label_changes': [],
        'confidence_changes': [],
        'summary': {
            'total_improvements': 0,
            'total_degradations': 0,
            'average_confidence_changes': 0
        }
    }
    #track matched detections
    matched_orig = [None] * len(orig_dets)
    matched_bright = [None] * len(bright_dets)
    highestSpatial = [None] * len(bright_dets)

    #find matching detections
    for i, orig in enumerate(orig_dets):
        best_match = None
        best_match_index = None
        best_similarity = spatial_threshold

        for j, bright in enumerate(bright_dets):
            similarity = calculate_spatial_similarity(orig, bright)
            print(f"spatial similarity between {orig.label} and {bright.label} is: ", similarity)
            if highestSpatial[j] is None:
                if similarity > best_similarity or (similarity >= best_similarity - 0.01 and bright.label == orig.label):
                    best_similarity = similarity
                    best_match = bright
                    best_match_index = j
            if highestSpatial[j] is not None:
                print("previous highestSpatial is :", highestSpatial[j])
                if similarity >= highestSpatial[j] - 0.01 and bright.label == orig.label:#change previously set
                    prevMatched = matched_bright[j]#yellow
                    prevIndex = orig_dets.index(prevMatched) #index of yellow
                    matched_orig[prevIndex] = None
                    best_match = bright
                    best_match_index = j
        if best_match is not None:
            matched_orig[i] = best_match
            matched_bright[best_match_index] = orig
            highestSpatial[best_match_index] = best_similarity
    for j, bright in enumerate(bright_dets):
        print("current best match is: ", matched_bright[j])
        if matched_bright[j] is not None:
            orig = matched_bright[j]
            print("original label: ", orig.label)
            print("brightened label: ", bright.label)
            #check for label changes
            if orig.label != bright.label:
                print('label change detected')
                results['label_changes'].append({
                    'original': {
                        'label': orig.label,
                        'confidence': orig.confidence,
                        'position': (orig.x, orig.y, orig.width, orig.height)
                    },
                    'brightened': {
                        'label': bright.label,
                        'confidence': bright.confidence,
                        'position': (bright.x, bright.y, bright.width, bright.height)
                    },
                    'spatial_similarity': highestSpatial[j]
                })
            else:
                #check for confidence changes
                #label remains the same
                confidence_change = bright.confidence - orig.confidence
                #care here as change = 0 is still appended in
                results['confidence_changes'].append({
                    'label': orig.label,
                    'original_confidence': orig.confidence,
                    'new_confidence': bright.confidence,
                    'change': confidence_change,
                    'position': (orig.x, orig.y, orig.width, orig.height),
                    'spatial_similarity': highestSpatial[j]
                })
    print("matched Bright: ", matched_bright)
    print("matched original: ", matched_orig)
    #case1: new detections (check for bugs here)
    for i, bright in enumerate(bright_dets):
        if matched_bright[i] is None:
            results['new_detections'].append({
                'label': bright.label,
                'confidence': bright.confidence,
                'position': (bright.x, bright.y, bright.width, bright.height)
            })
    #case2: lost detections
    for i, orig in enumerate(orig_dets):
        if matched_orig[i] is None: #original detection does not have matching detection
            results['lost_detections'].append({
                'label': orig.label,
                'confidence': orig.confidence,
                'position': (orig.x, orig.y, orig.width, orig.height)
            })

    #calculate summary statistics
    confidence_changes = [change['change'] for change in results['confidence_changes']]
    
    results['summary'].update({
        'total_improvements': len([c for c in confidence_changes if c > 0]),
        'total_degradations': len([c for c in confidence_changes if c < 0]),
        'total_confidence_changes': float(np.round(np.sum(confidence_changes), 3)) if confidence_changes else 0,
        'average_confidence_changes' : float(np.round(np.mean(confidence_changes), 3)) if confidence_changes else 0,
        'new_detections': len(results['new_detections']),
        'lost_detections': len(results['lost_detections']),
        'label_changes': len(results['label_changes']),
    })
    print("The summary results are: ", results)
    for change in results['confidence_changes']:
        change['change'] = float(round(change['change'], 2))
        change['spatial_similarity'] = float(np.round(change['spatial_similarity'], 2))
    return jsonify(results)

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
