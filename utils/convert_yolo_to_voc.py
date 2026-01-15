import os
import glob
import xml.etree.cElementTree as ET
from PIL import Image

# --- CONFIGURATION ---
# Class names must match your YOLO classes.txt order exactly
CLASSES = ['ABB', 'Kapital Bank', 'Pasha Bank']

def create_xml_file(image_path, boxes, output_path):
    """
    Creates a Pascal VOC XML file from a list of bounding boxes.
    
    Args:
        image_path (str): Path to the source image.
        boxes (list): List of [class_name, xmin, ymin, xmax, ymax].
        output_path (str): Path to save the XML file.
    """
    # 1. Read Image Dimensions
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            depth = 3 # 3 for RGB
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return

    # 2. Build XML Structure
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
    
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    # 3. Add Objects
    for box in boxes:
        class_name, xmin, ymin, xmax, ymax = box
        
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = class_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    # 4. Save formatted XML
    tree = ET.ElementTree(annotation)
    tree.write(output_path, encoding='utf-8')

def convert_dataset():
    """
    Iterates through train/test/val folders and converts YOLO labels to XML.
    """
    # Define project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    splits = ['train', 'val', 'test']
    
    print("Starting YOLO to Pascal VOC conversion...")

    for split in splits:
        img_dir = os.path.join(base_dir, "dataset", "images", split)
        lbl_dir = os.path.join(base_dir, "dataset", "labels", split)
        xml_dir = os.path.join(base_dir, "dataset", "xml_labels", split)

        os.makedirs(xml_dir, exist_ok=True)
        
        # Get all images
        image_files = glob.glob(os.path.join(img_dir, "*"))
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        converted_count = 0
        
        for img_path in image_files:
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(lbl_dir, file_name + ".txt")
            xml_path = os.path.join(xml_dir, file_name + ".xml")

            if not os.path.exists(txt_path):
                # Create empty XML for negative samples (background images)
                create_xml_file(img_path, [], xml_path)
                continue

            boxes = []
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])

                        # YOLO Normalized -> Absolute Pixels
                        xmin = int((x_center - w / 2) * img_w)
                        ymin = int((y_center - h / 2) * img_h)
                        xmax = int((x_center + w / 2) * img_w)
                        ymax = int((y_center + h / 2) * img_h)
                        
                        # Clip coordinates to image boundaries
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(img_w, xmax)
                        ymax = min(img_h, ymax)

                        if class_id < len(CLASSES):
                            boxes.append([CLASSES[class_id], xmin, ymin, xmax, ymax])

            create_xml_file(img_path, boxes, xml_path)
            converted_count += 1

        print(f"✅ [{split.upper()}] Converted {converted_count} files to {xml_dir}")

if __name__ == "__main__":
    convert_dataset()