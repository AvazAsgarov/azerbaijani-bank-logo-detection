import os
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- CONFIGURATION ---
# Define paths relative to the script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "images", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "results")

# Define all 6 Models
MODELS_CONFIG = {
    # YOLOv8
    "YOLOv8n_Baseline": os.path.join(BASE_DIR, "models", "yolo_v8", "yolov8n_baseline.pt"),
    "YOLOv8s_Baseline": os.path.join(BASE_DIR, "models", "yolo_v8", "yolov8s_baseline.pt"),
    "YOLOv8m_Augmented": os.path.join(BASE_DIR, "models", "yolo_v8", "yolov8m_augmented.pt"),
    
    # YOLOv5
    "YOLOv5s_Baseline": os.path.join(BASE_DIR, "models", "yolo_v5", "yolov5s_baseline.pt"),
    "YOLOv5m_Augmented": os.path.join(BASE_DIR, "models", "yolo_v5", "yolov5m_augmented.pt"),
    
    # Faster R-CNN
    "Faster_RCNN": os.path.join(BASE_DIR, "models", "faster_rcnn", "faster_rcnn_best.pth"),
}

# Faster R-CNN Class Map (0=Background, 1=ABB, 2=Kapital, 3=Pasha)
FRCNN_CLASSES = {1: 'ABB', 2: 'Kapital Bank', 3: 'Pasha Bank'}

# --- MODEL LOADING HELPERS ---

def load_frcnn_model(path):
    """Loads Faster R-CNN architecture and weights."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Re-build model (4 classes)
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
    
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model, device
    else:
        print(f"❌ Error: Faster R-CNN weights not found at {path}")
        return None, None

# --- INFERENCE FUNCTIONS ---

def predict_yolo(model, image_path):
    """Runs YOLO inference and formats results."""
    # conf=0.25 is standard for valid detection
    results = model.predict(image_path, verbose=False, conf=0.25)
    predictions = []
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist() # x1, y1, x2, y2
        
        # YOLOv5/v8 store names inside the model object
        label = model.names[cls_id]
        
        predictions.append({
            "label": label,
            "score": conf,
            "bbox": xyxy
        })
    return predictions

def predict_frcnn(model, device, image_path):
    """Runs Faster R-CNN inference and formats results."""
    img = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(img).to(device).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        
    predictions = []
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    
    for i in range(len(scores)):
        if scores[i] >= 0.25: # Confidence Threshold
            label_id = labels[i]
            if label_id in FRCNN_CLASSES:
                predictions.append({
                    "label": FRCNN_CLASSES[label_id],
                    "score": float(scores[i]),
                    "bbox": boxes[i].tolist()
                })
    return predictions

def draw_and_save(image_path, predictions, save_path):
    """Draws bounding boxes and saves the visualization."""
    img = cv2.imread(image_path)
    if img is None: return

    for p in predictions:
        x1, y1, x2, y2 = map(int, p['bbox'])
        label = f"{p['label']} {p['score']:.2f}"
        
        # Green Box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label Background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        
        # Text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

# --- MAIN EXECUTION ---

def main():
    print(f"🚀 Starting Batch Inference on Test Set...")
    print(f"📂 Images: {TEST_IMAGES_DIR}")
    print(f"💾 Output: {OUTPUT_DIR}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Gather Images
    test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not test_images:
        print("❌ No images found in test folder!")
        return

    all_results = []

    # 2. Iterate Models
    for model_name, model_path in MODELS_CONFIG.items():
        print(f"\n🔵 Processing: {model_name}...")
        
        # Load Model
        model = None
        device = None
        
        if "YOLO" in model_name:
            if os.path.exists(model_path):
                model = YOLO(model_path)
            else:
                print(f"   ⚠️ File not found: {model_path}")
                continue
        elif "Faster_RCNN" in model_name:
            model, device = load_frcnn_model(model_path)
            if model is None: continue
            
        # Run Inference Loop
        for img_file in tqdm(test_images, desc=f"   Inferring"):
            img_path = os.path.join(TEST_IMAGES_DIR, img_file)
            
            # Predict
            preds = []
            try:
                if "YOLO" in model_name:
                    preds = predict_yolo(model, img_path)
                else:
                    preds = predict_frcnn(model, device, img_path)
            except Exception as e:
                print(f"   ❌ Error on {img_file}: {e}")
                continue
            
            # Store Result
            all_results.append({
                "image_id": img_file,
                "model": model_name,
                "predictions": preds
            })
            
            # Save Visualization
            viz_path = os.path.join(OUTPUT_DIR, "visualizations", model_name, img_file)
            draw_and_save(img_path, preds, viz_path)

    # 3. Save Final JSON
    json_path = os.path.join(OUTPUT_DIR, "all_predictions.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\n✅ All inference complete.")
    print(f"📄 Raw Data saved to: {json_path}")
    print(f"🖼️ Visualizations saved to: {os.path.join(OUTPUT_DIR, 'visualizations')}")

if __name__ == "__main__":
    main()