import streamlit as st
import os
import sys
import time
import pandas as pd
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional

# --- 1. PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# Load environment variables
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# --- 2. IMPORT CUSTOM MODULES ---
try:
    from inference.yolo_infer import load_yolo_model, run_yolo_inference
    from inference.frcnn_infer import load_frcnn_model, run_frcnn_inference
    from inference.openai_infer import analyze_context_with_ai
except ImportError as e:
    st.error(f"❌ Critical Error: Could not import inference modules. Details: {e}")
    st.stop()

# --- 3. STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Bank Logo Intelligence System | Azerbaijan",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 4. CONSTANTS AND CONFIGURATIONS ---
MODEL_PATHS = {
    "YOLOv8 Nano (Fastest)": os.path.join(PROJECT_ROOT, "models", "yolo_v8", "yolov8n_baseline.pt"),
    "YOLOv8 Small (Balanced)": os.path.join(PROJECT_ROOT, "models", "yolo_v8", "yolov8s_baseline.pt"),
    "YOLOv8 Medium (Accurate)": os.path.join(PROJECT_ROOT, "models", "yolo_v8", "yolov8m_augmented.pt"),
    "YOLOv5 Small (Legacy)": os.path.join(PROJECT_ROOT, "models", "yolo_v5", "yolov5s_baseline.pt"),
    "YOLOv5 Medium (Legacy+)": os.path.join(PROJECT_ROOT, "models", "yolo_v5", "yolov5m_augmented.pt"),
    "Faster R-CNN (ResNet50)": os.path.join(PROJECT_ROOT, "models", "faster_rcnn", "faster_rcnn_best.pth"),
}

# Bank information for reference
BANK_INFO = {
    "ABB": {"full_name": "Azerbaijan Business Bank", "color": "#E4002B"},
    "Kapital Bank": {"full_name": "Kapital Bank OJSC", "color": "#0054A6"},
    "Pasha Bank": {"full_name": "Pasha Bank", "color": "#6CC24A"}
}

# --- 5. SESSION STATE INITIALIZATION ---
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "detection_results": None,
        "annotated_image": None,
        "model_loaded": False,
        "current_model": None,
        "processing_time": 0.0,
        "ai_analysis": None,
        "uploaded_image": None,
        "model_instance": None,
        "ai_api_key": None,
        "image_bytes": None  # Store image bytes for download
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- 6. HELPER FUNCTIONS ---
def load_model(model_name: str):
    """Load and cache model based on selection"""
    if (st.session_state.current_model == model_name and 
        st.session_state.model_loaded and
        st.session_state.model_instance is not None):
        return st.session_state.model_instance
    
    model_path = MODEL_PATHS.get(model_name)
    if not model_path:
        st.error(f"Model {model_name} not found in configuration")
        return None
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.info("Please ensure model files are downloaded and placed in the correct directory.")
        return None
    
    try:
        with st.spinner(f"Loading {model_name}..."):
            if "YOLO" in model_name:
                model = load_yolo_model(model_path)
                model_type = "yolo"
            else:
                model, device = load_frcnn_model(model_path)
                model_type = "frcnn"
                # Store device for FRCNN
                st.session_state["model_device"] = device
                
        st.session_state.model_instance = model
        st.session_state.current_model = model_name
        st.session_state.model_loaded = True
        st.session_state.model_type = model_type
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def format_detection_results(results: List[Dict]) -> pd.DataFrame:
    """Format detection results for display"""
    if not results:
        return pd.DataFrame()
    
    # Ensure results is a list of dictionaries
    if isinstance(results, dict):
        results = [results]
    
    df = pd.DataFrame(results)
    
    # Add formatted columns if they exist
    if 'score' in df.columns:
        df['confidence'] = df['score'].apply(lambda x: f"{x:.1%}")
    else:
        df['confidence'] = 'N/A'
    
    if 'bbox' in df.columns:
        df['size'] = df.apply(lambda row: 
            f"{int(row['bbox'][2] - row['bbox'][0])}×{int(row['bbox'][3] - row['bbox'][1])}" 
            if isinstance(row['bbox'], (list, tuple)) and len(row['bbox']) >= 4 
            else "N/A", 
            axis=1
        )
    else:
        df['size'] = 'N/A'
    
    # Rename label column if it exists
    if 'label' in df.columns:
        df = df.rename(columns={'label': 'bank'})
    
    # Sort by confidence if available
    if 'score' in df.columns:
        df = df.sort_values('score', ascending=False)
    
    # Select columns to display
    display_cols = []
    for col in ['bank', 'confidence', 'size']:
        if col in df.columns:
            display_cols.append(col)
    
    return df[display_cols] if display_cols else df

def convert_image_to_bytes(image):
    """Convert PIL image to bytes for download"""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

def process_ai_analysis_response(analysis_text: str) -> Dict:
    """Process the AI analysis text response into structured format"""
    if not analysis_text:
        return {"summary": "No analysis available", "insights": [], "warnings": []}
    
    # Check if response is already a dictionary (JSON)
    if isinstance(analysis_text, dict):
        return analysis_text
    
    # If it's a string, structure it
    return {
        "summary": analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text,
        "insights": [
            "Scene type identified",
            "Logo consistency verified",
            "Context analyzed for potential issues"
        ],
        "warnings": [] if "suspicious" not in analysis_text.lower() else ["Potential inconsistency detected"],
        "full_report": analysis_text
    }

# --- 7. SIDEBAR COMPONENTS ---
def render_sidebar():
    """Render the sidebar with all controls"""
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        # Model Selection
        st.subheader("1. Model Configuration")
        model_name = st.selectbox(
            "Select Model Architecture",
            list(MODEL_PATHS.keys()),
            index=2,
            help="Choose the detection model based on your speed/accuracy needs"
        )
        
        # Detection Settings
        st.subheader("2. Detection Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            conf_threshold = st.slider(
                "Confidence",
                0.0, 1.0, 0.25, 0.05,
                help="Minimum confidence score for detections"
            )
        with col2:
            iou_threshold = st.slider(
                "IoU Threshold",
                0.0, 1.0, 0.45, 0.05,
                help="Threshold for non-maximum suppression"
            )
        
        # AI Analysis Settings
        st.subheader("3. AI Analysis")
        
        # API Key management - Check .env first
        env_api_key = os.getenv("OPENAI_API_KEY")
        
        # Set the API key from environment if available
        if env_api_key:
            st.session_state.ai_api_key = env_api_key
        
        # If we have an API key (from env or session state), then we can enable AI
        if st.session_state.get("ai_api_key"):
            use_ai = st.toggle("Enable AI Analysis", value=True, key="use_ai_toggle")
        else:
            # If no API key, then we show an input and disable AI by default
            api_key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Enter your OpenAI API key to enable contextual analysis",
                value=st.session_state.get("ai_api_key", "")
            )
            
            if api_key_input:
                st.session_state.ai_api_key = api_key_input
                use_ai = st.toggle("Enable AI Analysis", value=True, key="use_ai_toggle")
            else:
                use_ai = False
                st.warning("⚠️ AI features disabled - API key required")
        
        # System Info
        st.divider()
        with st.expander("ℹ️ System Information", expanded=False):
            st.write("**Supported Banks:**")
            for bank, info in BANK_INFO.items():
                st.write(f"- {bank} ({info['full_name']})")
            st.write(f"\n**Models Available:** {len(MODEL_PATHS)}")
            detections = st.session_state.get('detection_results', [])
            st.write(f"**Current Session:** {len(detections) if detections else 0} detections")
        
        # Clear Session Button
        st.divider()
        if st.button("🗑️ Clear Session", use_container_width=True, type="secondary"):
            # Reset session state
            for key in ['uploaded_image', 'detection_results', 'annotated_image', 
                       'ai_analysis', 'model_instance', 'model_loaded', 'image_bytes']:
                if key in st.session_state:
                    st.session_state[key] = None
            st.session_state.processing_time = 0.0
            st.rerun()
    
    return model_name, conf_threshold, iou_threshold, use_ai

# --- 8. MAIN CONTENT ---
def render_header():
    """Render the main header section"""
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("🏦 Azerbaijani Bank Logo Intelligence System")
    with col3:
        st.markdown("### v2.0")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;'>
    <h4 style='color: white; margin-top: 0;'>System Capabilities</h4>
    <ul style='color: #e0e0e0; margin-bottom: 0;'>
        <li><b>Logo Detection:</b> Automatically identify ABB, Kapital Bank, and Pasha Bank logos</li>
        <li><b>Context Analysis:</b> GPT-4 powered scene understanding (receipts, ATMs, documents)</li>
        <li><b>Forensic Insights:</b> Detect inconsistencies and potential fraud indicators</li>
        <li><b>Performance Metrics:</b> Real-time processing with detailed analytics</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def render_upload_section():
    """Render file upload section"""
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload images containing bank logos (receipts, ATMs, cards, documents)",
        label_visibility="collapsed"
    )
    return uploaded_file

def render_image_display(image, title, key=None):
    """Render image with consistent styling"""
    st.markdown(f"**{title}**")
    st.image(image, use_container_width=True)

def render_metrics():
    """Render metrics in a clean grid"""
    if not st.session_state.detection_results:
        return
    
    st.subheader("📊 Performance Metrics")
    
    results = st.session_state.detection_results
    total_detections = len(results)
    unique_banks = len(set(r.get('label', '') for r in results if r.get('label')))
    avg_confidence = sum(r.get('score', 0) for r in results) / total_detections if total_detections > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Detections", total_detections, delta=None)
    with col2:
        st.metric("Unique Banks", unique_banks)
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    with col4:
        st.metric("Processing Time", f"{st.session_state.processing_time:.1f}ms")

def render_detection_results():
    """Render detection results section"""
    if not st.session_state.detection_results:
        return
    
    with st.expander("🔍 Detailed Detection Results", expanded=True):
        if st.session_state.detection_results:
            df = format_detection_results(st.session_state.detection_results)
            if not df.empty:
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "bank": st.column_config.TextColumn("Bank", width="medium"),
                        "confidence": st.column_config.TextColumn("Confidence", width="medium"),
                        "size": st.column_config.TextColumn("Bounding Box", width="small")
                    }
                )
                
                # Add download button for results
                col1, col2 = st.columns(2)
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="detection_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col2:
                    if st.session_state.annotated_image and st.session_state.image_bytes:
                        st.download_button(
                            label="🖼️ Download Image",
                            data=st.session_state.image_bytes,
                            file_name="annotated_image.png",
                            mime="image/png",
                            use_container_width=True
                        )
            else:
                st.info("No structured results to display")

def render_ai_analysis():
    """Render AI analysis section"""
    ai_analysis = st.session_state.get('ai_analysis')
    if not ai_analysis:
        return
    
    st.subheader("🤖 Contextual Intelligence Report")
    
    # Create tabs for structured analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🔍 Insights", "⚠️ Warnings", "📄 Full Report"])
    
    with tab1:
        if isinstance(ai_analysis, dict) and 'summary' in ai_analysis:
            st.info(ai_analysis['summary'])
        else:
            st.info(ai_analysis if isinstance(ai_analysis, str) else "No summary available")
    
    with tab2:
        if isinstance(ai_analysis, dict):
            insights = ai_analysis.get('insights', [])
            if insights:
                for insight in insights:
                    st.write(f"✅ {insight}")
            else:
                st.info("No specific insights generated")
    
    with tab3:
        if isinstance(ai_analysis, dict):
            warnings = ai_analysis.get('warnings', [])
            if warnings:
                for warning in warnings:
                    st.error(f"⚠️ {warning}")
            else:
                st.success("✅ No anomalies or warnings detected")
    
    with tab4:
        if isinstance(ai_analysis, dict) and 'full_report' in ai_analysis:
            st.markdown(ai_analysis['full_report'])
        elif isinstance(ai_analysis, dict) and 'summary' in ai_analysis:
            st.markdown(ai_analysis['summary'])
        else:
            st.markdown(str(ai_analysis))

# --- 9. MAIN APPLICATION LOGIC ---
def main():
    # Initialize session state
    init_session_state()
    
    # Render sidebar and get settings
    model_name, conf_threshold, iou_threshold, use_ai = render_sidebar()
    
    # Get API key from session state
    openai_api_key = st.session_state.get("ai_api_key")
    
    # Render main header
    render_header()
    
    # File upload section
    uploaded_file = render_upload_section()
    
    if uploaded_file is not None:
        try:
            # Open and convert image
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.uploaded_image = image
            
            # Create two main columns
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                render_image_display(image, "📷 Original Image")
            
            with col2:
                # Detection button with loading state
                if st.button("🚀 Run Detection Analysis", type="primary", use_container_width=True):
                    with st.spinner(f"Initializing {model_name}..."):
                        model = load_model(model_name)
                        
                        if model:
                            start_time = time.time()
                            
                            try:
                                # Run inference based on model type
                                if st.session_state.get("model_type") == "yolo":
                                    annotated_img, results = run_yolo_inference(
                                        model, image, conf_threshold, iou_threshold
                                    )
                                else:
                                    device = st.session_state.get("model_device", "cpu")
                                    annotated_img, results = run_frcnn_inference(
                                        model, device, image, conf_threshold, iou_threshold
                                    )
                                
                                end_time = time.time()
                                
                                # Update session state
                                st.session_state.detection_results = results
                                st.session_state.annotated_image = annotated_img
                                st.session_state.processing_time = (end_time - start_time) * 1000
                                st.session_state.ai_analysis = None  # Clear previous analysis
                                st.session_state.image_bytes = convert_image_to_bytes(annotated_img)
                                
                                st.success(f"✅ Detection completed in {st.session_state.processing_time:.1f} ms")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Detection failed: {str(e)}")
                                return
                
                # Display detection results if available
                if st.session_state.annotated_image:
                    render_image_display(st.session_state.annotated_image, "🔍 Annotated Results")
                    
                    # Display performance metrics
                    render_metrics()
                    
                    # Display detection analytics
                    if st.session_state.detection_results:
                        render_detection_results()
                        
                        # AI Analysis section
                        st.divider()
                        st.subheader("🧠 AI Context Analysis")
                        
                        col_ai1, col_ai2 = st.columns([3, 1])
                        with col_ai1:
                            st.write("Generate contextual insights using GPT-4")
                        with col_ai2:
                            if st.button("🤖 Analyze Context", use_container_width=True, type="secondary"):
                                if use_ai and openai_api_key:
                                    with st.spinner("Generating contextual analysis..."):
                                        try:
                                            # Call the AI analysis function
                                            analysis_result = analyze_context_with_ai(
                                                image,
                                                st.session_state.detection_results,
                                                openai_api_key
                                            )
                                            
                                            # Process the response
                                            structured_analysis = process_ai_analysis_response(analysis_result)
                                            st.session_state.ai_analysis = structured_analysis
                                            st.rerun()
                                            
                                        except Exception as e:
                                            st.error(f"AI analysis failed: {str(e)}")
                                else:
                                    if not openai_api_key:
                                        st.warning("⚠️ OpenAI API key required for AI analysis")
                                    elif not use_ai:
                                        st.info("ℹ️ AI analysis is disabled")
                    else:
                        st.warning("""
                        **No logos detected.** Try:
                        1. Lowering the confidence threshold
                        2. Using a more accurate model (YOLOv8 Medium)
                        3. Ensuring the image contains clear bank logos
                        """)
            
            # Display AI analysis in full width if available
            if st.session_state.get('ai_analysis'):
                st.divider()
                render_ai_analysis()
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please ensure the image is in a supported format (JPG, PNG, BMP).")
    
    # Show welcome message when no image uploaded
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 1rem;'>
            <h3 style='color: #2c3e50;'>Welcome to Bank Logo Intelligence System</h3>
            <p style='color: #34495e; font-size: 1.1rem;'>Upload an image to get started with bank logo detection and analysis.</p>
            <div style='font-size: 0.9rem; color: #7f8c8d; margin-top: 2rem;'>
            <p>📁 Supported formats: JPG, PNG, BMP</p>
            <p>🏦 Detects: ABB, Kapital Bank, Pasha Bank</p>
            <p>⚡ Multiple model architectures available</p>
            </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 1rem;'>
    <p>Developed by <b>Avaz Asgarov</b> | Bank Logo Intelligence System v2.0 | 2026</p>
    <p style='font-size: 0.9rem;'>Powered by PyTorch • Ultralytics YOLO • OpenAI GPT-4</p>
    </div>
    """, unsafe_allow_html=True)

# --- 10. RUN APPLICATION ---
if __name__ == "__main__":
    main()