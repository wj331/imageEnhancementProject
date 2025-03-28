import os
import random
import shutil
from mean_average_precision import MetricBuilder
import numpy as np

def generateRandom():
    exDarkRootImages ="C:/Users/wenji/OneDrive/Desktop/Y3S2/ATAP/sample images/ExDark/ExDark_images"
    exDarkRootAnnotations = "C:/Users/wenji/OneDrive/Desktop/Y3S2/ATAP/sample images/ExDark/ExDark_Annno"
    outputImageFolder = "C:/Users/wenji/OneDrive/Desktop/Y3S2/ATAP/sample images/ExDark/random_images"
    outputAnnotationsFolder = "C:/Users/wenji/OneDrive/Desktop/Y3S2/ATAP/sample images/ExDark/random_annotations"
    samplesPerClass = 5

    random.seed(42)
    categories = [d for d in os.listdir(exDarkRootImages) if os.path.isdir(os.path.join(exDarkRootImages, d))]

    for category in categories:
        category_path = os.path.join(exDarkRootImages, category)
        image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        sampledImages = random.sample(image_files, samplesPerClass)

        for img_name in sampledImages:
            src_path = os.path.join(category_path, img_name)
            dst_path = os.path.join(outputImageFolder, img_name)

            shutil.copy(src_path, dst_path)

            src_path_annotations = os.path.join(exDarkRootAnnotations, category, img_name + '.txt')

            dst_path_annotations = os.path.join(outputAnnotationsFolder, img_name + '.txt')
            shutil.copy(src_path_annotations, dst_path_annotations)

def scaleBackDetections(detections, scale_x, scale_y):
    """
    Scale back the detections to the original image size.
    Args:
        detections (list): List of detection objects with attributes x, y, width, height.
        scale_x (float): Scaling factor for x-axis.
        scale_y (float): Scaling factor for y-axis.
    Returns:
        list: List of scaled detection objects.
    """
    for det in detections:
        det['x'] = int(det['x'] * scale_x)
        det['y'] = int(det['y'] * scale_y)
        det['width'] = int(det['width'] * scale_x)
        det['height'] = int(det['height'] * scale_y)
    return detections

def compute_precision(ground_truths, predictions, iou_threshold=0.5):
    class_map = {
            "Bicycle": 0, "Boat": 1, "Bottle": 2, "Bus": 3,
            "Car": 4, "Cat": 5, "Chair": 6, "Cup": 7,
            "Dog": 8, "Motorbike": 9, "People": 10, "Table": 11
    }

    gts = []
    for gt in ground_truths:
        class_name = gt['label'].capitalize()
        if class_name not in class_map:
            continue
        class_id = class_map[class_name]
        gts.append([class_id, gt['x'], gt['y'], gt['width'], gt['height'], 1.0]) #dummy confidence 1.0 for GT
    print("ground truth formatted: ", gts)

    preds = []
    for det in predictions:
        predicted_class = det.label.capitalize()
        if predicted_class not in class_map:
            continue
        class_id = class_map[predicted_class]
        preds.append([class_id, det.x, det.y, det.width, det.height, det.confidence])
    print("predictions formatted: ", preds)

    # Initialize the metric
    true_positives = 0
    false_positives = 0
    used_gt = set()

    preds.sort(key=lambda x: x[5], reverse=True)

    for pred in preds:
        best_iou = iou_threshold
        best_gt_index = -1

        for i, gt in enumerate(gts):
            if i in used_gt:
                continue

            if pred[0] == gt[0]:
                iou = calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = i
        if  best_gt_index != -1:
            true_positives += 1
            used_gt.add(best_gt_index)
        else:
            false_positives += 1
        
    total_predicitons = len(preds)
    precision = true_positives / total_predicitons if total_predicitons > 0 else 0
    return precision
            
    

def calculate_iou(pred, gt):
    pred_class, x1_pred, y1_pred, w_pred, h_pred, pred_confidence = pred
    ground_class, x1_gt, y1_gt, w_gt, h_gt, ground_confidence = gt

    x2_pred = x1_pred + w_pred
    y2_pred = y1_pred + h_pred
    x2_gt = x1_gt + w_gt
    y2_gt = y1_gt + h_gt

    intersection_x1 = max(x1_pred, x1_gt)
    intersection_y1 = max(y1_pred, y1_gt)
    intersection_x2 = min(x2_pred, x2_gt)
    intersection_y2 = min(y2_pred, y2_gt)

    if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        pred_area = w_pred * h_pred
        gt_area = w_gt * h_gt
        union_area = pred_area + gt_area - intersection_area
        iou = intersection_area / union_area
        return iou
    else:
        return 0.0


