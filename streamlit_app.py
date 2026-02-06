import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="YOLO Plant Doctor", page_icon="🌿", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        color: #856404;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #155724;
        color: #155724;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        st.error("❌ Error: 'best.pt' file not found! Please upload it to GitHub.")
        return None
    return YOLO(model_path)

model = load_model()

# --- MAIN APP ---
st.title("🌿 Advanced Plant Disease Detector (YOLOv8)")
st.write("தக்காளி, உருளைக்கிழங்கு, மிளகாய் செடிகளில் உள்ள நோய்களைத் துல்லியமாக கண்டறியும் AI.")

uploaded_file = st.file_uploader("இலையின் படத்தை இங்கே பதிவேற்றவும்...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('🔍 Scan Now'):
        if model:
            with st.spinner('Scanning with YOLO AI...'):
                # YOLO Prediction
                results = model(image)
                
                # --- RESULT CHECKING ---
                # பாக்ஸ் (Boxes) எதுவும் இல்லை என்றால் -> நோய்/இலை இல்லை
                if len(results[0].boxes) == 0:
                    st.markdown("""
                    <div class="warning-box">
                        ⚠️ <b>எந்த நோயும் கண்டுபிடிக்கப்படவில்லை!</b><br>
                        1. இது ஆரோக்கியமான இலையாக இருக்கலாம்.<br>
                        2. அல்லது மாடலுக்குத் தெரியாத இலையாக இருக்கலாம் (Out of Scope).<br>
                        3. படம் தெளிவாக உள்ளதா எனப் பார்க்கவும்.
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    # பாக்ஸ் போடப்பட்ட படத்தை உருவாக்குதல் & காட்டுதல்
                    res_plotted = results[0].plot()
                    st.markdown('<div class="success-box">✅ குறைபாடுகள் கண்டறியப்பட்டன!</div>', unsafe_allow_html=True)
                    st.image(res_plotted, caption="AI Detection Result", use_column_width=True)
                    
                    # விவரங்கள்
                    st.write("---")
                    st.subheader("📋 கண்டறியப்பட்ட விவரங்கள்:")
                    for box in results[0].boxes:
                        class_name = model.names[int(box.cls[0])]
                        conf = float(box.conf[0]) * 100
                        st.write(f"🦠 **{class_name}** - {conf:.1f}% Confidence")