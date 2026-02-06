import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# 1. Page Configuration (Must be the first line)
st.set_page_config(
    page_title="AgroAI - Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Modern UI
st.markdown("""
    <style>
    /* Main Background with Soft Gradient */
    .stApp {
        background: linear-gradient(to right, #e0f7fa, #e8f5e9);
    }
    
    /* Header Style */
    .main-title {
        font-size: 3rem;
        color: #2e7d32;
        text-align: center;
        font-weight: bold;
        text-shadow: 2px 2px 4px #cccccc;
        margin-bottom: 20px;
    }
    
    /* Subheader */
    .sub-title {
        font-size: 1.5rem;
        color: #455a64;
        text-align: center;
        margin-bottom: 40px;
    }

    /* Result Card Style */
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        text-align: center;
        margin-top: 20px;
    }
    
    /* Success Text */
    .success-text {
        color: #2e7d32;
        font-size: 24px;
        font-weight: bold;
    }
    
    /* Warning Text */
    .warning-text {
        color: #d32f2f;
        font-size: 20px;
        font-weight: bold;
    }
    
    /* Button Style */
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar Content
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.title("AgroAI Control Panel")
    st.info("This AI tool helps farmers detect diseases in Tomato, Potato, and Pepper plants instantly.")
    st.markdown("---")
    confidence_threshold = st.slider("🤖 AI Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
    st.write("Higher value = More strict detection.")

# 4. Main Page Content
st.markdown('<div class="main-title">🌿 Intelligent Plant Disease Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a leaf image to get instant diagnosis and cure suggestions.</div>', unsafe_allow_html=True)

# 5. Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt') # Ensure best.pt is in the same folder

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# 6. File Uploader
uploaded_file = st.file_uploader("📂 Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Layout: Two Columns (Left: Image, Right: Results)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📸 Your Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True, caption="Uploaded Leaf")

    with col2:
        st.markdown("### 🩺 AI Diagnosis")
        
        # Add a spinner while processing
        with st.spinner('🔍 Analyzing leaf patterns...'):
            # Convert to format suitable for YOLO
            img_array = np.array(image)
            results = model.predict(img_array, conf=confidence_threshold)

            # --- PREDICTION LOGIC ---
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                disease_name = model.names[class_id]
                
                # --- SAFETY FILTER LOGIC (Strict Mode) ---
                # "Potato" என வந்தால் "Tomato" என மாற்றும் லாஜிக் (தேவைப்பட்டால் மட்டும்)
                if "Potato" in disease_name and "Tomato" not in disease_name:
                    # பயனர் தக்காளி என்று நினைத்தால் இதை மாற்றிக்கொள்ளலாம்
                    pass 

                # Display Result in a nice card
                st.markdown(f"""
                <div class="result-card">
                    <p style="color:gray;">Detected Issue:</p>
                    <p class="success-text">{disease_name.upper()}</p>
                    <p>Confidence: <b>{conf*100:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

                # Show Cure/Solution (Example Logic)
                st.markdown("### 💊 Recommended Cure:")
                if "blight" in disease_name.lower():
                    st.info("Use **Fungicides** like Mancozeb or Chlorothalonil. Remove infected leaves immediately.")
                elif "healthy" in disease_name.lower():
                    st.success("The plant looks healthy! Keep maintaining good watering habits. ✅")
                else:
                    st.warning("Consult a local agriculture expert for specific pesticide recommendations.")
                
            else:
                st.markdown("""
                <div class="result-card">
                    <p class="warning-text">⚠️ No Disease Detected</p>
                    <p>Try uploading a clearer image or moving closer to the leaf.</p>
                </div>
                """, unsafe_allow_html=True)
