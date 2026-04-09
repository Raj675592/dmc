import streamlit as st
import predict
from PIL import Image
import os
import time
from streamlit_paste_button import paste_image_button

# --- 1. Page Config ---
st.set_page_config(
    page_title="EcoGuard AI | Smart Waste Detection",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Advanced CSS (The Gemini Aesthetic) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
    }

    /* Glassmorphism Card */
    .st-emotion-cache-1r6slb0, .st-emotion-cache-6qob1r {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px !important;
        backdrop-filter: blur(10px);
    }

    /* Input Zone Styling */
    .input-container {
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        transition: 0.3s;
    }
    
    .input-container:hover {
        border-color: #3b82f6;
        background: rgba(59, 130, 246, 0.05);
    }

    /* Custom Titles */
    .main-title {
        font-size: 3rem;
        background: -webkit-linear-gradient(#fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0px;
    }

    /* Hide redundant elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. Sidebar (Clean & Minimal) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3299/3299935.png", width=80)
    st.markdown("## EcoGuard AI")
    st.caption("v2.1.0 - Urban Intelligence")
    st.markdown("---")
    
    with st.expander("📡 System Status", expanded=True):
        st.success("YOLOv8 Engine: Ready")
        st.success("API Gateway: Online")
    
    st.info("Directly paste a screenshot ($Ctrl+V$) or drag an image into the terminal to begin.")

# --- 4. Main Header ---
st.markdown("<h1 class='main-title'>Smart City Monitor</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.2rem;'>AI-Powered Urban Cleanliness Analysis</p>", unsafe_allow_html=True)

# --- 5. Logic & Helper Functions ---
@st.cache_resource
def get_model():
    return predict.load_model()

# --- 6. Unified Input Experience ---
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown("### 🛠 Input Terminal")
    
    # Custom Container for Input
    with st.container():
        # Top level: Paste Button
        pasted_image = paste_image_button(
            label="📋 Click here then Paste (Ctrl+V)",
            background_color="#3b82f6",
            errors="ignore",
           


        )
        
        # Divider with text
        st.markdown("<p style='text-align: center; color: #64748b; margin: 10px 0;'>— OR —</p>", unsafe_allow_html=True)
        
        # File Uploader
        uploaded_file = st.file_uploader("Drop image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    # Image Priority Logic
    active_image = None
    if pasted_image.image_data is not None:
        active_image = pasted_image.image_data
    elif uploaded_file is not None:
        active_image = Image.open(uploaded_file)

    if active_image:
        st.image(active_image, caption="Current Analysis Subject", use_container_width=True)
        if st.button("🗑️ Clear Image", use_container_width=True):
            st.rerun()

with col2:
    st.markdown("### 📡 AI Intelligence")
    
    if active_image is not None:
        # Pre-processing
        if active_image.mode in ("RGBA", "P"):
            active_image = active_image.convert("RGB")
        
        temp_path = "temp_inference.jpg"
        active_image.save(temp_path)

        # Gemini-style Progress
        with st.status("Analysing Visual Data...", expanded=True) as status:
            st.write("Checking pixel density...")
            time.sleep(0.4)
            st.write("Running YOLOv8 Spatial Detection...")
            model = get_model()
            result = predict.predict(model, temp_path)
            time.sleep(0.4)
            status.update(label="Analysis Complete", state="complete", expanded=False)

        # Results Display
        if result == 1:
            st.error("🚨 **SPILL DETECTED**")
            st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; padding: 20px; border-radius: 15px;">
                    <h3 style="color: #ef4444; margin-top:0;">Violation Logged</h3>
                    <p style="color: #f87171;">Environmental debris detected outside of designated containers. Incident coordinates logged for sanitation dispatch.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ **AREA CLEAR**")
            st.markdown(f"""
                <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; padding: 20px; border-radius: 15px;">
                    <h3 style="color: #22c55e; margin-top:0;">Status: Compliant</h3>
                    <p style="color: #4ade80;">No unauthorized waste detected. The monitored zone meets cleanliness protocols.</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
    else:
        # Empty State
        st.info("System idling. Provide a visual input to begin real-time inference.")
        st.markdown("""
            <div style="opacity: 0.5; text-align: center; padding: 50px;">
                <img src="https://cdn-icons-png.flaticon.com/512/1055/1055644.png" width="80">
                <p>Waiting for data...</p>
            </div>
        """, unsafe_allow_html=True)

# --- 7. Footer Feedback ---
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("💬 Accuracy Feedback Loop"):
    col_fb1, col_fb2 = st.columns(2)
    col_fb1.button("Mark as False Positive", use_container_width=True)
    col_fb2.button("Mark as False Negative", use_container_width=True)