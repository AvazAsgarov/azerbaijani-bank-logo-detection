import os
import json
import glob
import time
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_JSON = os.path.join(BASE_DIR, "reports", "results", "all_predictions.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "results")
DATA_YAML = os.path.join(BASE_DIR, "dataset", "data.yaml")
TEST_IMGS_DIR = os.path.join(BASE_DIR, "dataset", "images", "test")
TEST_LABELS_DIR = os.path.join(BASE_DIR, "dataset", "labels", "test")

# IOU Threshold for defining a "Correct" detection
IOU_THRESHOLD = 0.5 

MODELS_CONFIG = {
    "YOLOv8n_Baseline": os.path.join(BASE_DIR, "models", "yolo_v8", "yolov8n_baseline.pt"),
    "YOLOv8s_Baseline": os.path.join(BASE_DIR, "models", "yolo_v8", "yolov8s_baseline.pt"),
    "YOLOv8m_Augmented": os.path.join(BASE_DIR, "models", "yolo_v8", "yolov8m_augmented.pt"),
    "YOLOv5s_Baseline": os.path.join(BASE_DIR, "models", "yolo_v5", "yolov5s_baseline.pt"),
    "YOLOv5m_Augmented": os.path.join(BASE_DIR, "models", "yolo_v5", "yolov5m_augmented.pt"),
    "Faster_RCNN": os.path.join(BASE_DIR, "models", "faster_rcnn", "faster_rcnn_best.pth"),
}

# --- HELPER FUNCTIONS ---

def compute_iou(boxA, boxB):
    """Calculates IoU between two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def load_ground_truth():
    """Reads all test labels into a dictionary."""
    gt = {}
    print("📂 Loading Ground Truth Labels...")
    for label_file in glob.glob(os.path.join(TEST_LABELS_DIR, "*.txt")):
        # Get matching image name
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        # Find image to get dimensions (for un-normalizing)
        img_path = os.path.join(TEST_IMGS_DIR, base_name + ".jpg")
        if not os.path.exists(img_path): img_path = img_path.replace(".jpg", ".png")
        
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                w, h = img.size
        else:
            w, h = 640, 640 # Default fallback

        boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                c, x, y, bw, bh = map(float, line.split())
                # Un-normalize YOLO format
                x1 = (x - bw/2) * w
                y1 = (y - bh/2) * h
                x2 = (x + bw/2) * w
                y2 = (y + bh/2) * h
                
                # Class map (0=ABB, 1=Kapital, 2=Pasha) -> matches your YAML
                cls_name = ['ABB', 'Kapital Bank', 'Pasha Bank'][int(c)]
                boxes.append({'label': cls_name, 'bbox': [x1, y1, x2, y2]})
        
        gt[base_name] = boxes
    return gt

def calculate_manual_metrics(model_preds, gt_data):
    """
    Compares predictions vs GT to get TP, FP, FN, P, R, F1.
    """
    tp, fp, fn = 0, 0, 0
    
    # Track which GT boxes were matched to avoid double counting
    for img_id, items in model_preds.items():
        base_name = os.path.splitext(img_id)[0]
        ground_truths = gt_data.get(base_name, [])
        predictions = items # List of dicts
        
        matched_gt = set()
        
        # Sort predictions by confidence (high to low)
        predictions.sort(key=lambda x: x['score'], reverse=True)
        
        for p in predictions:
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching GT
            for i, gt in enumerate(ground_truths):
                if gt['label'] == p['label']:
                    iou = compute_iou(p['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
            
            if best_iou >= IOU_THRESHOLD and best_gt_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
                
        # Any GT not matched is a False Negative
        fn += len(ground_truths) - len(matched_gt)

    # Calculate metrics
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4)
    }

def get_yolo_official_stats(model_path):
    """Runs YOLO val() to get mAP and Speed."""
    try:
        model = YOLO(model_path)
        metrics = model.val(data=DATA_YAML, split='test', verbose=False)
        return {
            "mAP@50": round(metrics.box.map50, 4),
            "mAP@50-95": round(metrics.box.map, 4),
            "Speed (ms)": round(metrics.speed['inference'], 2)
        }
    except:
        return {"mAP@50": 0, "mAP@50-95": 0, "Speed (ms)": 0}

def measure_frcnn_speed(model_path):
    """Manually measures inference speed for Faster R-CNN."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fasterrcnn_resnet50_fpn(weights=None)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 4)
    
    if not os.path.exists(model_path): return 0
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    # Warmup
    dummy = torch.rand(1, 3, 640, 640).to(device)
    model(dummy)
    
    times = []
    images = glob.glob(os.path.join(TEST_IMGS_DIR, "*"))[:10] # Measure on 10 images
    for img_p in images:
        img = Image.open(img_p).convert("RGB")
        t_img = T.ToTensor()(img).to(device).unsqueeze(0)
        t0 = time.time()
        model(t_img)
        times.append((time.time() - t0) * 1000)
        
    return round(np.mean(times), 2)

# --- MAIN ---

def main():
    print("🚀 Starting Complete Evaluation (Manual + Official)...")
    
    # 1. Load Ground Truth
    gt_data = load_ground_truth()
    
    # 2. Load Predictions
    with open(PREDICTIONS_JSON, 'r') as f:
        all_preds = json.load(f)
        
    # Group preds by model
    preds_by_model = {}
    for p in all_preds:
        m = p['model']
        if m not in preds_by_model: preds_by_model[m] = {}
        preds_by_model[m][p['image_id']] = p['predictions']

    final_results = []

    # 3. Process Each Model
    for model_name, model_path in MODELS_CONFIG.items():
        print(f"\n🔵 Evaluating: {model_name}...")
        
        # A. Calculate Manual Classification Metrics (P, R, F1)
        if model_name in preds_by_model:
            manual_metrics = calculate_manual_metrics(preds_by_model[model_name], gt_data)
        else:
            manual_metrics = {"TP":0, "FP":0, "FN":0, "Precision":0, "Recall":0, "F1-Score":0}
            
        # B. Get Official Detection Metrics (mAP, Speed)
        official_metrics = {}
        if "YOLO" in model_name:
            official_metrics = get_yolo_official_stats(model_path)
        else:
            # Faster R-CNN
            speed = measure_frcnn_speed(model_path)
            official_metrics = {
                "mAP@50": "N/A", 
                "mAP@50-95": "N/A", 
                "Speed (ms)": speed
            }
            
        # C. Combine
        row = {"Model": model_name}
        row.update(official_metrics)
        row.update(manual_metrics)
        final_results.append(row)

    # 4. Save
    df = pd.DataFrame(final_results)
    
    # Reorder columns
    cols = ["Model", "Speed (ms)", "Precision", "Recall", "F1-Score", "mAP@50", "mAP@50-95", "TP", "FP", "FN"]
    df = df[cols]
    
    csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Done! Saved full metrics to: {csv_path}")
    print(df)

if __name__ == "__main__":
    main()