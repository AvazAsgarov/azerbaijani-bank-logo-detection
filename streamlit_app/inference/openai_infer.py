import base64
import io
import streamlit as st
from PIL import Image
from openai import OpenAI

def encode_image_to_base64(image):
    """
    Converts a PIL Image to a Base64 encoded string.
    Required for passing images to the OpenAI API.
    """
    buffered = io.BytesIO()
    # Save as JPEG to keep payload size reasonable
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_context_with_ai(image, detections, api_key):
    """
    Uses GPT-4o to analyze the semantic context of the image based on object detections.
    
    Args:
        image (PIL.Image): The input image.
        detections (list): List of dicts [{'label': 'ABB', 'score': 0.95}, ...].
        api_key (str): OpenAI API Key.
        
    Returns:
        str: The AI's analysis in Markdown format.
    """
    if not api_key:
        return "⚠️ Error: No API Key provided. Please enter your key in the sidebar."

    # 1. Prepare Data
    try:
        base64_image = encode_image_to_base64(image)
        
        # Summarize detections for the LLM
        if detections:
            det_summary = ", ".join([f"{d['label']} ({d['score']:.1%})" for d in detections])
        else:
            det_summary = "No specific bank logos were detected by the object detector."

        # 2. Define the System Prompt (Prompt Engineering)
        # Persona: Forensic Analyst specialized in Azerbaijani Banking
        # Task: Scene Classification, OCR extraction, and Context Verification
        system_prompt = """
        You are a Senior Financial Forensics Analyst & Computer Vision Specialist for the Azerbaijani banking sector.
        Your goal is to analyze an image to determine the context of a banking transaction or presence.
        
        You will receive:
        1. An image containing a bank logo (ABB, Kapital Bank, or Pasha Bank).
        2. A list of logos already detected by a YOLO model.

        Your Output must be a structured Markdown analysis covering:
        
        ### 1. 📍 Scene Classification
        Classify the image into one of these categories:
        - **Transaction Receipt/Slip** (Paper proof)
        - **ATM/Terminal** (Screen or Physical machine)
        - **Branch/Storefront** (Physical building)
        - **Digital Interface** (Mobile app or Website screenshot)
        - **Marketing/Billboard** (Advertisement)
        - **Other/Unknown**

        ### 2. 📝 Data Extraction (OCR)
        If text is visible, extract the following (otherwise state "N/A"):
        - **Transaction Date/Time:** (e.g., 15.01.2025 14:30)
        - **Total Amount:** (Look for AZN, ₼, or USD)
        - **Merchant/Location Name:**

        ### 3. 🔍 Contextual Verification
        - **Consistency Check:** Does the detected logo match the text in the image? (e.g., if YOLO found 'Kapital Bank', does the receipt text also say 'Kapital Bank'?)
        - **Anomaly Detection:** Does the image look fraudulent, photoshopped, or is the ATM screen showing an error?

        ### 4. 💡 Executive Summary
        A one-sentence conclusion about what is happening in this image.
        """

        # 3. Call OpenAI API
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o", # Use gpt-4o or gpt-4-turbo
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": f"The object detection model found these logos: {det_summary}. Please analyze the image context."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.3 # Low temperature for factual analysis
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"❌ OpenAI API Error: {str(e)}"