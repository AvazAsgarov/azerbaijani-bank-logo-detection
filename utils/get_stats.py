import os
import torch
import time
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

# --- HELPER: Load Faster R-CNN Architecture ---
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def load_frcnn(model_path, num_classes=4):
    """Loads the Faster R-CNN model architecture and weights."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Recreate the architecture (ResNet50 + FPN)
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model, device
    else:
        print(f"❌ Error: Model not found at {model_path}")
        return None, None

def check_yolo_stats(model_name, model_path, data_yaml_path):
    """Uses Ultralytics native validation to print metrics."""
    if not os.path.exists(model_path):
        print(f"⚠️ Skipping {model_name} (File not found)")
        return

    print(f"\n📊 Checking Stats for: {model_name}")
    try:
        model = YOLO(model_path)
        # Validate on the 'test' split
        metrics = model.val(data=data_yaml_path, split='test', verbose=False)
        
        print(f"   ► mAP@50:    {metrics.box.map50:.4f}")
        print(f"   ► mAP@50-95: {metrics.box.map:.4f}")
        print(f"   ► Precision: {metrics.box.mp:.4f}")
        print(f"   ► Recall:    {metrics.box.mr:.4f}")
        print(f"   ► Speed:     {metrics.speed['inference']:.2f} ms/img")
        
    except Exception as e:
        print(f"   ❌ Error validating {model_name}: {e}")

def check_frcnn_stats(model_name, model_path, test_img_dir):
    """Checks inference speed for Faster R-CNN."""
    if not os.path.exists(model_path):
        print(f"⚠️ Skipping {model_name} (File not found)")
        return

    print(f"\n📊 Checking Stats for: {model_name}")
    model, device = load_frcnn(model_path)
    if not model: return

    # Get images
    images = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not images:
        print("   ⚠️ No images found in test folder.")
        return

    # Warmup GPU
    if device.type == 'cuda':
        dummy = torch.rand(1, 3, 640, 640).to(device)
        model(dummy)

    # Timing Loop
    times = []
    print(f"   Running inference on {len(images)} images for speed check...")
    
    with torch.no_grad():
        for img_name in tqdm(images, unit="img"):
            img_path = os.path.join(test_img_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            
            # Preprocess
            img_t = T.ToTensor()(img).to(device).unsqueeze(0)
            
            # Infer
            t0 = time.time()
            model(img_t)
            t1 = time.time()
            times.append((t1 - t0) * 1000)

    avg_time = np.mean(times)
    print(f"   ► Avg Inference Speed: {avg_time:.2f} ms/img")
    print("   ► mAP/Precision: (Check 'evaluate_models.py' output for these)")

def main():
    # Define Project Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml = os.path.join(base_dir, "dataset", "data.yaml")
    test_dir = os.path.join(base_dir, "dataset", "images", "test")
    
    # Define All 6 Models
    models_config = {
        # YOLOv8
        "YOLOv8n (Baseline)": os.path.join(base_dir, "models", "yolo_v8", "yolov8n_baseline.pt"),
        "YOLOv8s (Baseline)": os.path.join(base_dir, "models", "yolo_v8", "yolov8s_baseline.pt"),
        "YOLOv8m (Augmented)": os.path.join(base_dir, "models", "yolo_v8", "yolov8m_augmented.pt"),
        
        # YOLOv5
        "YOLOv5s (Baseline)": os.path.join(base_dir, "models", "yolo_v5", "yolov5s_baseline.pt"),
        "YOLOv5m (Augmented)": os.path.join(base_dir, "models", "yolo_v5", "yolov5m_augmented.pt"),
        
        # Faster R-CNN
        "Faster R-CNN": os.path.join(base_dir, "models", "faster_rcnn", "faster_rcnn_best.pth"),
    }

    print("🚀 Starting Stats Check for all 6 Models...")

    for name, path in models_config.items():
        if "Faster R-CNN" in name:
            check_frcnn_stats(name, path, test_dir)
        else:
            # Assume YOLO
            check_yolo_stats(name, path, data_yaml)

    print("\n✅ Stats check complete.")

if __name__ == "__main__":
    main()