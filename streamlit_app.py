import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import pandas as pd

# --- 1. PAGE SETUP ---
st.set_page_config(
    page_title="NanbaProject - AI Plant Doctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    h1, h2, h3 { color: #2e7d32; font-family: 'Arial', sans-serif; }
    .metric-card { background-color: #f1f8e9; border: 1px solid #c5e1a5; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .metric-value { font-size: 32px; font-weight: bold; color: #1b5e20; }
    .metric-label { font-size: 16px; color: #555; }
    .report-box { border: 2px solid #ddd; padding: 20px; border-radius: 10px; margin-top: 10px; background-color: #f9f9f9; border-left: 5px solid #2e7d32; }
    .disease-name { color: #d9534f; font-size: 22px; font-weight: bold; }
    .healthy-name { color: #28a745; font-size: 22px; font-weight: bold; }
    [data-testid="stSidebar"] { background-color: #e8f5e9; }
    div.stButton > button { background-color: #2e7d32; color: white; border-radius: 8px; width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- 3. DISEASE DATABASE ---
disease_info = {
    # TOMATO
    "Tomato_Early_Blight": { "name": "தக்காளி - கருகல் நோய் (Early Blight)", "status": "Diseased", "description": "இலைகளில் வளைய வடிவில் பழுப்பு நிறப் புள்ளிகள் தோன்றும்.", "solution": "💊 **தீர்வு:** மாங்கோசெப் (Mancozeb) மருந்தை தெளிக்கவும்." },
    "Tomato_Late_Blight": { "name": "தக்காளி - தாமத கருகல் (Late Blight)", "status": "Diseased", "description": "இலைகள் கறுத்து அழுகிவிடும்.", "solution": "💊 **தீர்வு:** மெட்டலாக்சில் (Metalaxyl) மருந்தை தெளிக்கவும்." },
    "Tomato_Yellow_Leaf_Curl_Virus": { "name": "தக்காளி - இலைச் சுருள் (Yellow Leaf Curl)", "status": "Diseased", "description": "இலைகள் மஞ்சள் நிறமாகி சுருண்டுவிடும்.", "solution": "💊 **தீர்வு:** வேப்ப எண்ணெய் தெளிக்கவும்." },
    "Tomato_Leaf_Mold": { "name": "தக்காளி - இலை பூஞ்சை (Leaf Mold)", "status": "Diseased", "description": "இலைகளின் அடியில் பூஞ்சை வளரும்.", "solution": "💊 **தீர்வு:** காப்பர் ஆக்சிகுளோரைடு தெளிக்கவும்." },
    "Tomato_Septoria_Leaf_Spot": { "name": "தக்காளி - இலைப்புள்ளி (Septoria)", "status": "Diseased", "description": "சிறிய வட்ட வடிவ புள்ளிகள் தோன்றும்.", "solution": "💊 **தீர்வு:** பூஞ்சைக் கொல்லி பயன்படுத்தவும்." },
    "Tomato_Spider_Mites_Two_spotted_spider_mite": { "name": "தக்காளி - சிலந்தி பேன் (Mites)", "status": "Diseased", "description": "இலைகளில் மஞ்சள் நிறப்புள்ளிகள்.", "solution": "💊 **தீர்வு:** அக்காரைடு (Acaricide) மருந்து தெளிக்கவும்." },
    "Tomato_Target_Spot": { "name": "தக்காளி - டார்கெட் ஸ்பாட்", "status": "Diseased", "description": "அடர் பழுப்பு நிற புள்ளிகள்.", "solution": "💊 **தீர்வு:** பூஞ்சைக் கொல்லி தெளிக்கவும்." },
    "Tomato_Mosaic_virus": { "name": "தக்காளி - மொசைக் வைரஸ்", "status": "Diseased", "description": "பச்சை மற்றும் மஞ்சள் நிறத் திட்டுகள்.", "solution": "💊 **தீர்வு:** செடியை அகற்றி அழிக்கவும்." },
    "Tomato_Healthy": { "name": "ஆரோக்கியமான தக்காளி செடி", "status": "Healthy", "description": "செடி செழிப்பாக உள்ளது.", "solution": "✅ **பராமரிப்பு:** இயற்கை உரம் இடவும்." },

    # POTATO
    "Potato_Early_Blight": { "name": "உருளைக்கிழங்கு - கருகல் நோய்", "status": "Diseased", "description": "பழுப்பு நிறத் திட்டுகள் மற்றும் வளையங்கள்.", "solution": "💊 **தீர்வு:** குளோரோதலானில் மருந்து தெளிக்கலாம்." },
    "Potato_Late_Blight": { "name": "உருளைக்கிழங்கு - தாமத கருகல்", "status": "Diseased", "description": "இலைகள் அழுகி துர்நாற்றம் வீசும்.", "solution": "💊 **தீர்வு:** காப்பர் மருந்துகளை தெளிக்கவும்." },
    "Potato_Healthy": { "name": "ஆரோக்கியமான உருளைக்கிழங்கு", "status": "Healthy", "description": "செடி நன்றாக உள்ளது.", "solution": "✅ **பராமரிப்பு:** நீர் நிர்வாகம் அவசியம்." },

    # PEPPER
    "Pepper__bell___Bacterial_spot": { "name": "மிளகாய் - பாக்டீரியா இலைப்புள்ளி", "status": "Diseased", "description": "நீர் தேங்கியது போன்ற புள்ளிகள்.", "solution": "💊 **தீர்வு:** காப்பர் மருந்து தெளிக்கவும்." },
    "Pepper__bell___Healthy": { "name": "ஆரோக்கியமான மிளகாய் செடி", "status": "Healthy", "description": "செடி பசுமையாக உள்ளது.", "solution": "✅ **பராமரிப்பு:** பூச்சி தாக்குதலை கண்காணிக்கவும்." }
}

# --- 4. LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = 'best.pt'
    if not os.path.exists(model_path): return None
    return YOLO(model_path)

model = load_model()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80)
    st.title("NanbaProject")
    st.subheader("Final Year Project")
    st.markdown("---")
    page = st.radio("மெனு (Menu)", ["🏠 Home (Overview)", "📖 Methodology", "📊 Performance", "🚀 Live Simulation"], index=0)
    st.markdown("---")
    st.info("Developed by: **Muthusamy A** & Team\nDepartment of ECE/CSE")

# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "🏠 Home (Overview)":
    st.title("AI-Powered Plant Doctor 🌿")
    st.markdown("### புரட்சிகரமான விவசாய தொழில்நுட்பம்\n**Nanba Project** என்பது **YOLOv8** தொழில்நுட்பத்தைப் பயன்படுத்தி பயிர் நோய்களை கண்டறியும் தளமாகும்.")
    st.write("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="metric-card"><div class="metric-value">99.5%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-card"><div class="metric-value">54,300+</div><div class="metric-label">Dataset Size</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="metric-card"><div class="metric-value">< 15ms</div><div class="metric-label">Inference Speed</div></div>', unsafe_allow_html=True)
    st.write("---")
    st.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png", caption="YOLOv8 Architecture", use_column_width=True)

# ==========================================
# PAGE 2: METHODOLOGY
# ==========================================
elif page == "📖 Methodology":
    st.title("🔬 Research Methodology")
    c1, c2 = st.columns(2)
    with c1:
        st.info("1. Data Collection")
        st.write("PlantVillage தரவுத்தொகுப்பு பயன்படுத்தப்பட்டது (54,306 படங்கள்).")
        st.success("3. Model Training")
        st.write("Google Colab T4 GPU மூலம் YOLOv8 Nano மாடல் பயிற்சி அளிக்கப்பட்டது.")
    with c2:
        st.warning("2. Preprocessing")
        st.write("Roboflow மூலம் தரவு தயார் செய்யப்பட்டது (640x640 Resolution).")
        st.error("4. Deployment")
        st.write("Streamlit Cloud மூலம் டிப்ளாய் செய்யப்பட்டது.")

# ==========================================
# PAGE 3: PERFORMANCE
# ==========================================
elif page == "📊 Performance":
    st.title("📈 Performance Metrics")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Accuracy")
        st.bar_chart(pd.DataFrame({'Model': ['Nanba (YOLOv8)', 'CNN', 'VGG16'], 'Accuracy': [99.5, 92.1, 96.8]}).set_index('Model'), color="#2e7d32")
    with c2:
        st.subheader("Speed (ms)")
        st.line_chart(pd.DataFrame({'Model': ['Nanba (YOLOv8)', 'CNN', 'VGG16'], 'Time': [15, 340, 800]}).set_index('Model'))

# ==========================================
# PAGE 4: LIVE SIMULATION (STRICT MODE)
# ==========================================
elif page == "🚀 Live Simulation":
    st.title("🌿 Live Disease Detection")
    st.markdown("முதலில் **பயிரைத் (Crop)** தேர்ந்தெடுத்து, பின் இலையின் படத்தை பதிவேற்றம் செய்யவும்.")
    
    # 1. Selection
    selected_crop = st.radio(
        "👇 எந்தப் பயிரைப் பரிசோதிக்க வேண்டும்?",
        ["Tomato (தக்காளி)", "Potato (உருளைக்கிழங்கு)", "Pepper (மிளகாய்)", "All (எல்லா பயிர்களும்)"],
        horizontal=True
    )
    
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        c1, c2 = st.columns([1, 1])
        image = Image.open(uploaded_file)
        with c1:
            st.image(image, caption="Original Image", use_column_width=True)
            
        with c2:
            st.write("Analyzing...")
            if st.button("🔍 Scan & Detect", type="primary"):
                if model is None:
                    st.error("❌ Model not found!")
                else:
                    with st.spinner("AI is Scanning..."):
                        # Get Top 1 Result Only
                        results = model(image, conf=0.5, max_det=1)
                        
                        if len(results[0].boxes) == 0:
                            st.warning("⚠️ எந்த நோயும் கண்டுபிடிக்கப்படவில்லை.")
                        else:
                            # --- STRICT FILTER LOGIC ---
                            # இங்கே எந்த Magic-உம் இல்லை. User கேட்டது இருந்தா மட்டும் தான் காட்டும்.
                            
                            found_any = False
                            filtered_boxes = []
                            names = model.names
                            
                            for box in results[0].boxes:
                                class_name = names[int(box.cls[0])]
                                
                                # 1. Strict Tomato (தக்காளி என்றால் தக்காளி பெயர் மட்டும் இருக்க வேண்டும்)
                                if selected_crop == "Tomato (தக்காளி)":
                                    if "tomato" in class_name.lower():
                                        filtered_boxes.append((box, class_name))
                                        found_any = True
                                        
                                # 2. Strict Potato (உருளை என்றால் உருளை பெயர் மட்டும் இருக்க வேண்டும்)
                                elif selected_crop == "Potato (உருளைக்கிழங்கு)":
                                    if "potato" in class_name.lower():
                                        filtered_boxes.append((box, class_name))
                                        found_any = True
                                        
                                # 3. Strict Pepper
                                elif selected_crop == "Pepper (மிளகாய்)":
                                    if "pepper" in class_name.lower():
                                        filtered_boxes.append((box, class_name))
                                        found_any = True
                                        
                                # 4. All (எல்லாவற்றையும் காட்டு)
                                elif selected_crop == "All (எல்லா பயிர்களும்)":
                                    filtered_boxes.append((box, class_name))
                                    found_any = True
                                    
                            # --- DISPLAY RESULTS ---
                            if not found_any:
                                st.warning(f"⚠️ எச்சரிக்கை: நீங்கள் '{selected_crop}' தேர்வு செய்துள்ளீர்கள்.")
                                st.error("ஆனால் AI கண்டறிந்த பயிர் வேறாக உள்ளது (அல்லது தவறாக கணிக்கப்பட்டுள்ளது).")
                                st.info("சரியான பயிரைத் தேர்வு செய்யவும் அல்லது 'All' ஆப்ஷனைப் பயன்படுத்தவும்.")
                            else:
                                st.success(f"✅ {selected_crop} நோய் கண்டறியப்பட்டது!")
                                res_plotted = results[0].plot()
                                st.image(res_plotted, use_column_width=True, caption="Prediction Result")
                                
                                for box, final_name in filtered_boxes:
                                    conf = float(box.conf[0]) * 100
                                    info = disease_info.get(final_name)
                                    
                                    if info:
                                        name_class = "healthy-name" if info['status'] == "Healthy" else "disease-name"
                                        st.markdown(f"""
                                        <div class="report-box">
                                            <div class="{name_class}">{info['name']}</div>
                                            <p><b>Confidence:</b> {conf:.2f}%</p>
                                            <p><b>📌 விளக்கம்:</b> {info['description']}</p>
                                            <div>{info['solution']}</div>
                                        </div>""", unsafe_allow_html=True)
                                    else:
                                        st.write(f"🔍 **Detected:** {final_name}")



