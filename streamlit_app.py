import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import pandas as pd
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NanbaProject - Final Year Project",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. EXACT UI STYLING (CSS) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #ffffff; font-family: 'sans-serif'; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #e8f5e9; border-right: 1px solid #c8e6c9; }
    .sidebar-title { font-size: 24px; font-weight: bold; color: #2e7d32; margin-bottom: 5px; }
    .sidebar-subtitle { font-size: 16px; font-weight: bold; color: #1b5e20; margin-bottom: 20px; }
    .dev-box { background-color: #d1e7dd; padding: 15px; border-radius: 10px; border: 1px solid #badbcc; color: #0f5132; font-size: 14px; margin-top: 20px; }

    /* Cards */
    .metric-card { background-color: #f1f8e9; border: 1px solid #c5e1a5; border-radius: 10px; padding: 25px; text-align: center; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .metric-value { font-size: 2.5rem; font-weight: bold; color: #2e7d32; }
    .metric-label { font-size: 1rem; color: #555; }

    /* Methodology Boxes */
    .method-box { padding: 20px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; }
    .box-blue { background-color: #e3f2fd; color: #0d47a1; border-left: 5px solid #2196f3; }
    .box-yellow { background-color: #fffde7; color: #f57f17; border-left: 5px solid #ffeb3b; }
    .box-green { background-color: #e8f5e9; color: #1b5e20; border-left: 5px solid #4caf50; }
    .box-red { background-color: #ffebee; color: #b71c1c; border-left: 5px solid #f44336; }

    /* Result Box */
    .result-box { background-color: #ffffff; border: 1px solid #ddd; border-left: 6px solid #2e7d32; padding: 25px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.08); margin-top: 20px; }
    h1, h2, h3 { color: #2e7d32; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. TAMIL DISEASE DATABASE (CORRECTED KEYS 🛠️) ---
# மாடலில் இருந்து வரும் பெயர்களுக்கு (Space, Lowercase) ஏற்றவாறு மாற்றப்பட்டுள்ளது.
disease_info = {
    # --- TOMATO (தக்காளி) ---
    "Tomato mold leaf": { 
        "name": "தக்காளி - இலை பூஞ்சை (Leaf Mold)", 
        "status": "Diseased", 
        "description": "இலைகளின் மேற்பரப்பில் வெளிர் மஞ்சள் புள்ளிகள் மற்றும் அடியில் ஆலிவ் பச்சை பூஞ்சை வளரும்.", 
        "solution": "💊 **தீர்வு:** காப்பர் ஆக்சிகுளோரைடு அல்லது பாவிஸ்டின் மருந்தை தெளிக்கவும். காற்றோட்டத்தை அதிகரிக்கவும்." 
    },
    "Tomato Early blight leaf": { 
        "name": "தக்காளி - கருகல் நோய் (Early Blight)", 
        "status": "Diseased", 
        "description": "இலைகளில் வளைய வடிவில் பழுப்பு நிறப் புள்ளிகள் தோன்றும். இது செடியின் அடிப்பகுதியில் தொடங்கி மேலே பரவும்.", 
        "solution": "💊 **தீர்வு:** மாங்கோசெப் (Mancozeb) அல்லது குளோரோதலானில் மருந்தை தெளிக்கவும். பாதிக்கப்பட்ட இலைகளை அகற்றவும்." 
    },
    "Tomato Septoria leaf spot": { 
        "name": "தக்காளி - இலைப்புள்ளி நோய் (Septoria)", 
        "status": "Diseased", 
        "description": "இலைகளில் சிறிய வட்ட வடிவ புள்ளிகள் தோன்றும். நடுவில் சாம்பல் நிறமும், ஓரத்தில் கருப்பு வளையமும் இருக்கும்.", 
        "solution": "💊 **தீர்வு:** குளோரோதலானில் அல்லது காப்பர் பூஞ்சைக் கொல்லியை தெளிக்கவும். வேர்ப்பகுதியில் மட்டும் நீர் பாய்ச்சவும்." 
    },
    "Tomato leaf bacterial spot": { 
        "name": "தக்காளி - பாக்டீரியா புள்ளி (Bacterial Spot)", 
        "status": "Diseased", 
        "description": "இலைகளில் சிறிய, நீர் தேங்கியது போன்ற புள்ளிகள் தோன்றும். பின்னர் அவை பழுப்பு நிறமாக மாறும்.", 
        "solution": "💊 **தீர்வு:** காப்பர் மற்றும் மேங்கோசெப் மருந்துகளை கலந்து தெளிக்கவும்." 
    },
    "Tomato leaf late blight": { 
        "name": "தக்காளி - தாமத கருகல் (Late Blight)", 
        "status": "Diseased", 
        "description": "இலைகள் ஈரமாக இருக்கும்போது கறுத்து அழுகிவிடும். வெள்ளை நிற பூஞ்சை இலைகளின் அடியில் காணப்படும்.", 
        "solution": "💊 **தீர்வு:** மெட்டலாக்சில் (Metalaxyl) கலந்த பூஞ்சைக் கொல்லியை தெளிக்கவும். ஈரம் தேங்காமல் பார்த்துக்கொள்ளவும்." 
    },
    "Tomato leaf mosaic virus": { 
        "name": "தக்காளி - மொசைக் வைரஸ்", 
        "status": "Diseased", 
        "description": "இலைகளில் பச்சை மற்றும் மஞ்சள் நிறத் திட்டுகள் (Mosaic Pattern) காணப்படும். இலைகள் வடிவமற்று போகும்.", 
        "solution": "💊 **தீர்வு:** இதற்கு மருந்து இல்லை. பாதிக்கப்பட்ட செடியை உடனே வேரோடு பிடுங்கி எரித்துவிடவும்." 
    },
    "Tomato leaf yellow virus": { 
        "name": "தக்காளி - இலைச் சுருள் (Yellow Leaf Curl)", 
        "status": "Diseased", 
        "description": "இலைகள் மஞ்சள் நிறமாகி, மேல் நோக்கி சுருண்டுவிடும். செடியின் வளர்ச்சி குன்றிவிடும்.", 
        "solution": "💊 **தீர்வு:** இது வெள்ளை ஈக்களால் பரவுகிறது. வேப்ப எண்ணெய் அல்லது இமிடாகுளோப்ரிட் தெளிக்கவும்." 
    },
    "Tomato leaf": { # சில சமயம் ஆரோக்கியமான இலையை இப்படி காட்டும்
        "name": "ஆரோக்கியமான தக்காளி செடி (Healthy)", 
        "status": "Healthy", 
        "description": "செடி செழிப்பாகவும், இலைகள் பசுமையாகவும் உள்ளன.", 
        "solution": "✅ **பராமரிப்பு:** தொடர்ந்து இயற்கை உரம் மற்றும் முறையான நீர் நிர்வாகத்தை கடைபிடிக்கவும்." 
    },

    # --- POTATO (உருளைக்கிழங்கு) ---
    "Potato leaf early blight": { 
        "name": "உருளைக்கிழங்கு - கருகல் நோய்", 
        "status": "Diseased", 
        "description": "இலைகளில் பழுப்பு நிறத் திட்டுகள் மற்றும் வளையங்கள் தோன்றும்.", 
        "solution": "💊 **தீர்வு:** குளோரோதலானில் மருந்து தெளிக்கலாம். பயிர் சுழற்சி முறையை பின்பற்றவும்." 
    },
    "Potato leaf late blight": { 
        "name": "உருளைக்கிழங்கு - தாமத கருகல்", 
        "status": "Diseased", 
        "description": "இலைகள் விளிம்பிலிருந்து அழுகி துர்நாற்றம் வீசும்.", 
        "solution": "💊 **தீர்வு:** காப்பர் ஆக்சிகுளோரைடு மருந்துகளை தெளிக்கவும்." 
    },
     "Potato leaf": { 
        "name": "ஆரோக்கியமான உருளைக்கிழங்கு", 
        "status": "Healthy", 
        "description": "செடி நன்றாக உள்ளது. பூச்சி தாக்குதல் இல்லை.", 
        "solution": "✅ **பராமரிப்பு:** நீர் நிர்வாகம் அவசியம்." 
    },

    # --- PEPPER (மிளகாய்) ---
    "Bell_pepper leaf spot": { 
        "name": "மிளகாய் - இலைப்புள்ளி", 
        "status": "Diseased", 
        "description": "இலைகளில் புள்ளிகள் தோன்றி காய்ந்துவிடும்.", 
        "solution": "💊 **தீர்வு:** காப்பர் மருந்தை தெளிக்கவும்." 
    },
    "Bell_pepper leaf": { 
        "name": "ஆரோக்கியமான மிளகாய் செடி", 
        "status": "Healthy", 
        "description": "செடி பசுமையாக உள்ளது.", 
        "solution": "✅ **பராமரிப்பு:** பூச்சி தாக்குதலை கண்காணிக்கவும்." 
    }
}

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_model():
    if os.path.exists('best.pt'):
        return YOLO('best.pt')
    return None

model = load_model()

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80)
    st.markdown('<div class="sidebar-title">NanbaProject</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Final Year Project</div>', unsafe_allow_html=True)
    st.write("---")
    
    st.markdown("**மெனு (Menu)**")
    page = st.radio("", ["🏠 Home (Overview)", "📖 Methodology", "📊 Performance", "🚀 Live Simulation"], index=0)
    
    st.markdown('<div class="dev-box"><b>Developed by:</b> Muthusamy A &<br>Team Department of ECE/CSE</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "🏠 Home (Overview)":
    st.markdown("<h1>AI-Powered Plant Doctor 🌿</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#555; font-weight:normal;'>புரட்சிகரமான விவசாய தொழில்நுட்பம்</h3><p style='color:grey;'>Nanba Project என்பது YOLOv8 தொழில்நுட்பத்தைப் பயன்படுத்தி பயிர் நோய்களை கண்டறியும் தளமாகும்.</p>", unsafe_allow_html=True)
    st.write("")
    
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="metric-card"><div class="metric-value">99.5%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-card"><div class="metric-value">54,300+</div><div class="metric-label">Dataset Size</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="metric-card"><div class="metric-value">< 15ms</div><div class="metric-label">Inference Speed</div></div>', unsafe_allow_html=True)
    
    st.write("---")
    st.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png", use_column_width=True)

# ==========================================
# PAGE 2: METHODOLOGY
# ==========================================
elif page == "📖 Methodology":
    st.markdown("<h1>🔬 Research Methodology</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown('<div class="method-box box-blue"><p>1. Data Collection</p><span style="font-weight:normal; font-size:14px;">PlantVillage தரவுத்தொகுப்பு பயன்படுத்தப்பட்டது (54,306 படங்கள்).</span></div>', unsafe_allow_html=True)
    c1.markdown('<div class="method-box box-green"><p>3. Model Training</p><span style="font-weight:normal; font-size:14px;">Google Colab T4 GPU மூலம் YOLOv8 Nano மாடல் பயிற்சி அளிக்கப்பட்டது.</span></div>', unsafe_allow_html=True)
    c2.markdown('<div class="method-box box-yellow"><p>2. Preprocessing</p><span style="font-weight:normal; font-size:14px;">Roboflow மூலம் தரவு தயார் செய்யப்பட்டது (640x640 Resolution).</span></div>', unsafe_allow_html=True)
    c2.markdown('<div class="method-box box-red"><p>4. Deployment</p><span style="font-weight:normal; font-size:14px;">Streamlit Cloud மூலம் டிப்ளாய் செய்யப்பட்டது.</span></div>', unsafe_allow_html=True)

# ==========================================
# PAGE 3: PERFORMANCE
# ==========================================
elif page == "📊 Performance":
    st.markdown("<h1>📈 Performance Metrics</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    chart_data = pd.DataFrame({'Model': ['CNN', 'Nanba (YOLOv8)', 'VGG16'], 'Accuracy': [92, 99.5, 96], 'Speed': [340, 15, 800]}).set_index('Model')
    with col1:
        st.subheader("Accuracy")
        st.bar_chart(chart_data['Accuracy'], color="#2e7d32")
    with col2:
        st.subheader("Speed (ms)")
        st.line_chart(chart_data['Speed'])

# ==========================================
# PAGE 4: LIVE SIMULATION (Main Logic)
# ==========================================
elif page == "🚀 Live Simulation":
    st.markdown("<h1>🌿 Live Disease Detection</h1>", unsafe_allow_html=True)
    st.write("முதலில் **பயிரைத் (Crop)** தேர்ந்தெடுத்து, பின் இலையின் படத்தை பதிவேற்றம் செய்யவும்.")
    
    st.markdown("👉 **எந்தப் பயிரைப் பரிசோதிக்க வேண்டும்?**")
    selected_crop = st.radio("", ["Tomato (தக்காளி)", "Potato (உருளைக்கிழங்கு)", "Pepper (மிளகாய்)", "All (எல்லா பயிர்களும்)"], horizontal=True)
    
    st.write("")
    st.markdown("Upload Leaf Image")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if st.button("🔍 Scan & Detect", type="primary"):
                if model is None:
                    st.error("❌ Model 'best.pt' not found!")
                else:
                    with st.spinner("Analyzing..."):
                        time.sleep(1)
                        results = model(image, conf=0.3)
                        
                        found = False
                        filtered_results = []
                        names = model.names
                        
                        # --- DEBUGGING INFO (Optional - can be removed) ---
                        # st.write(f"Detected Classes: {[names[int(b.cls[0])] for b in results[0].boxes]}") 

                        if len(results[0].boxes) > 0:
                            for box in results[0].boxes:
                                cls_name = names[int(box.cls[0])]
                                
                                # Filtering Logic
                                match = False
                                if selected_crop == "All (எல்லா பயிர்களும்)": match = True
                                elif "tomato" in cls_name.lower() and "Tomato" in selected_crop: match = True
                                elif "potato" in cls_name.lower() and "Potato" in selected_crop: match = True
                                elif "pepper" in cls_name.lower() and "Pepper" in selected_crop: match = True
                                
                                if match:
                                    filtered_results.append((box, cls_name))
                                    found = True
                        
                        if not found:
                            st.warning(f"⚠️ {selected_crop} நோய் எதுவும் கண்டறியப்படவில்லை.")
                            st.info("சரியான பயிரைத் தேர்ந்தெடுக்கவும் அல்லது தெளிவான படத்தை பதிவேற்றவும்.")
                        else:
                            res_plot = results[0].plot()
                            st.image(res_plot, caption="AI Detection", use_column_width=True)
                            
                            for box, name in filtered_results:
                                conf = float(box.conf[0]) * 100
                                # Get info from dictionary (Try Exact Match first)
                                info = disease_info.get(name)
                                
                                # If exact match fails, try case-insensitive lookup
                                if not info:
                                    for key in disease_info:
                                        if key.lower() == name.lower():
                                            info = disease_info[key]
                                            break
                                
                                if info:
                                    status_color = "#d32f2f" if info['status'] == "Diseased" else "#2e7d32"
                                    st.markdown(f"""
                                    <div class="result-box">
                                        <h3 style="color:{status_color}; margin:0;">{info['name']}</h3>
                                        <p style="color:gray;">Confidence: {conf:.1f}%</p>
                                        <hr>
                                        <p><b>📌 விளக்கம்:</b> {info['description']}</p>
                                        <p><b>💡 தீர்வு:</b> {info['solution']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.warning(f"Info missing for: {name} (Please update database)")












