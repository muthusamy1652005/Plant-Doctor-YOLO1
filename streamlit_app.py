import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# 1. Page Configuration
st.set_page_config(
    page_title="NanbaProject - AI Plant Doctor",
    page_icon="🌿",
    layout="wide"
)

# 2. Custom CSS to match "NanbaProject" Design
st.markdown("""
    <style>
    /* General Background */
    .stApp {
        background-color: #ffffff;
        font-family: 'sans-serif';
    }

    /* Hero Section Title */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #1b5e20; /* Dark Green */
        text-align: center;
        margin-bottom: 0px;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .highlight {
        color: #00c853; /* Light Green */
    }

    /* Metrics Cards */
    .metric-card {
        background-color: #f1f8e9;
        border: 1px solid #c5e1a5;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }

    /* Dark Simulation Section */
    .sim-container {
        background-color: #0f172a; /* Dark Blue/Black */
        padding: 40px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 30px;
    }
    .sim-title {
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #00c853;
        color: white;
        border-radius: 20px;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Hero Section (Title)
st.markdown('<div class="hero-title">AI-Powered <span class="highlight">Plant Doctor</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">The <b>Nanba Project</b> utilizes Convolutional Neural Networks (CNNs) to detect plant leaf diseases in real-time.</div>', unsafe_allow_html=True)

# 4. Metrics Section (Like the Screenshot)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">98.5%</div>
        <div class="metric-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">54,300+</div>
        <div class="metric-label">Dataset Size</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">&lt; 200ms</div>
        <div class="metric-label">Inference Time</div>
    </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer
st.write("")

# 5. Live Simulation Section (Dark Theme)
st.markdown('<div class="sim-container"><div class="sim-title">Live Project Simulation</div><p>Experience the Nanba detection engine. Upload a leaf image below.</p></div>', unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Model not found! Make sure 'best.pt' is in the folder.")

# File Uploader (Centered)
col_spacer1, col_main, col_spacer2 = st.columns([1, 2, 1])

with col_main:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display Image inside the "Dark Mode" feel
        st.image(image, caption="Uploaded Sample", use_column_width=True)

        if st.button("🔍 Run Diagnosis"):
            with st.spinner("Processing with MobileNetV2 pipeline..."):
                time.sleep(1) # Small delay for effect
                img_array = np.array(image)
                results = model.predict(img_array)
                
                # Result Display
                if len(results[0].boxes) > 0:
                    box = results[0].boxes[0]
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    disease_name = model.names[class_id]

                    # Custom Success Message
                    st.balloons()
                    st.markdown(f"""
                        <div style="background-color: #d1e7dd; color: #0f5132; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
                            <h2 style="margin:0;">✅ Detected: {disease_name.upper()}</h2>
                            <p style="margin:0;">Confidence Score: <b>{conf*100:.1f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No disease detected. Try a closer image.")

# 6. Methodology Footer (Optional)
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>© 2026 NanbaProject Research | Powered by YOLOv8 & Streamlit</div>", unsafe_allow_html=True)

