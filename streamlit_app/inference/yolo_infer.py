import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

@st.cache_resource
def load_yolo_model(model_path):
    """
    Loads a YOLO model (v5 or v8) and caches it in memory.
    
    Args:
        model_path (str): Path to the .pt file.
        
    Returns:
        model: The loaded YOLO model object, or None if failed.
    """
    try:
        print(f"🔄 Loading YOLO model from: {model_path}...")
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {e}")
        return None

def run_yolo_inference(model, image, conf_threshold, iou_threshold):
    """
    Runs object detection on a single image.

    Args:
        model: Loaded YOLO model.
        image (PIL.Image): Input image.
        conf_threshold (float): Confidence threshold (0.0 - 1.0).
        iou_threshold (float): NMS IoU threshold (0.0 - 1.0).

    Returns:
        annotated_image (PIL.Image): Image with bounding boxes drawn.
        results_list (list): List of dicts with keys 'label', 'score', 'bbox'.
    """
    try:
        # Run inference
        # imgsz=640 is standard, but you can increase for small text
        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            verbose=False
        )
        
        # 1. Generate Visual Output
        # plot() returns a NumPy array in BGR format (Blue-Green-Red)
        plotted_array = results[0].plot()
        
        # Convert BGR to RGB for Streamlit/PIL
        plotted_array_rgb = plotted_array[..., ::-1]
        annotated_image = Image.fromarray(plotted_array_rgb)
        
        # 2. Extract Structured Data
        results_list = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            bbox = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            
            # Get class name
            label = model.names[cls_id]
            
            results_list.append({
                "label": label,
                "score": score,
                "bbox": bbox
            })
            
        return annotated_image, results_list

    except Exception as e:
        st.error(f"Error during YOLO inference: {e}")
        return image, []