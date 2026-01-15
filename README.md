# 🏦 Azerbaijani Bank Logo Detection & Forensic Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991)

**Real-time object detection system for verifying Azerbaijani bank logos (ABB, Kapital, Pasha), integrated with an AI-powered forensic dashboard for semantic scene analysis.**

---

## 📖 Project Overview

Financial fraud, including phishing sites and counterfeit receipts, is a growing threat in Azerbaijan's digital banking sector. This project provides an automated computer vision solution to detect and verify the presence of major bank logos. 

Moving beyond simple bounding boxes, this system introduces **Intelligent Forensic Analysis**: it combines the visual detection results with **GPT-4o** to "read" the context of the image, verifying if a detected logo matches the transaction text on a receipt or the screen of an ATM.

### 🎯 Supported Classes
* **ABB** (International Bank of Azerbaijan)
* **Kapital Bank**
* **Pasha Bank**

---

## 🚀 Key Features

* **Multi-Model Architecture:** Compare performance across **YOLOv5**, **YOLOv8** (Nano/Small/Medium), and **Faster R-CNN**.
* **Real-Time Dashboard:** A user-friendly web app built with **Streamlit** for instant image analysis.
* **Forensic AI Agent:** Uses **GPT-4o** to perform OCR, scene classification (e.g., "Is this a receipt?"), and consistency checks (e.g., "Does the receipt text match the detected logo?").
* **Comprehensive Reporting:** Automatically generates Excel reports with Confusion Matrices and precision/recall metrics.

---

## 📊 Model Performance

We trained and evaluated 6 different models. **YOLOv8n (Nano)** emerged as the best candidate for deployment due to its superior speed-accuracy balance.

| Model | Speed (ms) | Precision | Recall | F1-Score | mAP@50 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n (Best)** | **46.43** | **0.9167** | 0.9167 | **0.9167** | 0.9401 |
| YOLOv8s | 93.00 | 0.8684 | 0.9167 | 0.8919 | 0.9442 |
| YOLOv5m (Robust) | 219.33 | 0.8333 | **0.9722** | 0.8974 | **0.9736** |
| Faster R-CNN | 1099.41 | 0.6000 | 0.9167 | 0.7253 | N/A |

> *Note: Faster R-CNN suffered from high false positives (22 FP) due to background confusion.*

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/AzerbaijaniBankLogoDetection.git](https://github.com/your-username/AzerbaijaniBankLogoDetection.git)
    cd AzerbaijaniBankLogoDetection
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your OpenAI Key:
    ```env
    OPENAI_API_KEY=sk-your_api_key_here
    ```

---

## 🖥️ Usage

### 1. Run the Dashboard (Streamlit)
To launch the interactive web interface:
```bash
streamlit run streamlit_app/app.py

```

*Upload an image, select a model (e.g., YOLOv8n), and click "Run Object Detection".*

### 2. Run Batch Inference

To process a folder of images and generate a JSON report:

```bash
python scripts/run_inference.py

```

### 3. Generate Excel Report

To compile metrics and confusion matrices into an Excel file:

```bash
python scripts/generate_excel_report.py

```

---

## 📂 Project Structure

```text
├── dataset/                # Train/Val/Test images and labels
├── models/                 # Trained weights (.pt and .pth files)
├── notebooks/              # Google Colab training notebooks
├── reports/                # Generated Excel reports and Confusion Matrices
├── scripts/                # Python scripts for evaluation and inference
├── streamlit_app/          # Source code for the web dashboard
├── utils/                  # Helper functions (plotting, converters)
├── .env                    # API keys
└── requirements.txt        # Python dependencies

```

---

## 🔮 Future Work

* **Expand Dataset:** Include **Unibank** and **Leobank** classes.
* **Real-Time Video:** Adapt the pipeline for live RTSP stream processing.
* **Edge Deployment:** Quantize the YOLOv8n model for mobile deployment (TFLite/ONNX).

---

## 👨‍💻 Author

**Avaz Asgarov**

---

*This project was developed as a final capstone, demonstrating the complete machine learning lifecycle from data collection to deployment.*

```
