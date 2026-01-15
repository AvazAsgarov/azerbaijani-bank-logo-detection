import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- CONFIGURATION ---
# Faster R-CNN typically uses 0 for background.
# 1=ABB, 2=Kapital, 3=Pasha. Total 4 classes.
CLASSES = {
    0: 'Background',
    1: 'ABB',
    2: 'Kapital Bank',
    3: 'Pasha Bank'
}
NUM_CLASSES = 4 

@st.cache_resource
def load_frcnn_model(model_path):
    """
    Loads the Faster R-CNN model architecture and weights.
    Uses @st.cache_resource to avoid reloading on every interaction.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    try:
        # 1. Re-initialize the exact same architecture used in training
        # weights=None prevents downloading massive COCO weights unnecessarily
        model = fasterrcnn_resnet50_fpn(weights=None)
        
        # 2. Replace the head (Predictor) for our specific 4 classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        
        # 3. Load the trained weights
        print(f"🔄 Loading Faster R-CNN from: {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        
        # 4. Set to evaluation mode (Critical for inference!)
        model.to(device)
        model.eval()
        
        return model, device
        
    except Exception as e:
        st.error(f"❌ Error loading Faster R-CNN: {e}")
        return None, None

def run_frcnn_inference(model, device, image, conf_threshold, iou_threshold):
    """
    Runs inference, filters results, and draws boxes manually.

    Args:
        model: Loaded PyTorch model.
        device: 'cpu' or 'cuda'.
        image (PIL.Image): Input image.
        conf_threshold (float): Minimum score to show a box.
        iou_threshold (float): (Not used heavily here as FRCNN does internal NMS).

    Returns:
        annotated_image (PIL.Image): Image with boxes drawn.
        results_list (list): Structured data for the table.
    """
    try:
        # 1. Preprocess Image
        # Convert PIL to Tensor (0.0 - 1.0)
        transform = T.ToTensor()
        img_tensor = transform(image).to(device)
        
        # Add batch dimension [C, H, W] -> [1, C, H, W]
        img_tensor = img_tensor.unsqueeze(0)

        # 2. Run Inference
        with torch.no_grad():
            prediction = model(img_tensor)

        # 3. Process Results (Remove from GPU)
        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()

        # 4. Draw Boxes using OpenCV
        # Convert PIL image to numpy array (RGB) for drawing
        img_np = np.array(image)
        
        # Streamlit images are RGB, but OpenCV usually expects BGR for saving.
        # However, since we return to Streamlit/PIL, we keep it in RGB logic.
        # So drawing (0, 255, 0) means Green in standard RGB.

        results_list = []

        for i in range(len(scores)):
            score = scores[i]
            
            # Apply Confidence Threshold
            if score >= conf_threshold:
                box = boxes[i]
                label_id = labels[i]
                label_name = CLASSES.get(label_id, 'Unknown')
                
                # Coordinates
                x1, y1, x2, y2 = box.astype(int)
                
                # --- Drawing ---
                # 1. Rectangle (Green)
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 2. Label Background (Filled Green Box)
                text = f"{label_name}: {score:.2f}"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img_np, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
                
                # 3. Text (Black)
                cv2.putText(img_np, text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                # Store for Table
                results_list.append({
                    "label": label_name,
                    "score": float(score),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })

        # Convert back to PIL for Streamlit display
        annotated_image = Image.fromarray(img_np)
        
        return annotated_image, results_list

    except Exception as e:
        st.error(f"Error during Faster R-CNN inference: {e}")
        return image, []