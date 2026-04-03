import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from predict import load_model, predict, CLASS_NAMES, CLASS_INFO

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Alzheimer's Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #6C3483;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 0;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #6C3483;
        border-left: 4px solid #6C3483;
        padding-left: 10px;
        margin: 1.5rem 0 1rem 0;
    }
    .disclaimer {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #856404;
    }
    .stDownloadButton button {
        background-color: #6C3483 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model("model/alzheimer_model.h5")

model = get_model()

# ─── Header ───────────────────────────────────────────────────
st.markdown('<p class="main-title">🧠 Alzheimer\'s Disease Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered MRI brain scan analysis using Deep Learning</p>', unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("About This Tool")

    st.markdown("""
    This tool uses a **Deep Learning CNN** trained on **33,984 MRI brain scans** 
    to detect and classify Alzheimer's disease stages.
    """)

    st.subheader("📊 4 Detection Classes")
    for class_name, info in CLASS_INFO.items():
        st.markdown(f"{info['emoji']} **{info['label']}**")

    st.markdown("---")

    st.subheader("📈 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy",  "82%+")
    col2.metric("AUC-ROC",   "88%+")
    col1.metric("Model",     "EfficientNetB0")
    col2.metric("Dataset",   "33,984 MRI")

    st.markdown("---")

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Medical Disclaimer</b><br>
    This tool is for educational and research purposes only. 
    It is NOT a substitute for professional medical diagnosis. 
    Always consult a qualified neurologist for medical advice.
    </div>
    """, unsafe_allow_html=True)

# ─── Main Area ────────────────────────────────────────────────
if model is None:
    st.error("❌ Model not found! Please ensure `model/alzheimer_model.h5` exists.")
    st.info("""
    **To set up the model:**
    1. Train the model using `src/train.py`
    2. Place the trained `alzheimer_model.h5` in the `model/` folder
    3. Restart the app
    """)
    st.stop()

st.markdown('<p class="section-header">📤 Upload MRI Brain Scan</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload an MRI brain scan image",
        type=["jpg", "jpeg", "png"],
        help="Upload a top-view or axial MRI brain scan for best results"
    )

with col2:
    st.info("""
    **Tips for best results:**
    - Use axial (top-view) MRI brain scans
    - Clear, high resolution images work best
    - The model was trained on the Augmented Alzheimer MRI Dataset from Kaggle
    """)

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(pil_img, caption="Uploaded MRI Scan", use_container_width=True)

    with col2:
        with st.spinner("🧠 Analysing MRI scan..."):
            result = predict(model, pil_img)

        info       = result['class_info']
        pred_class = result['predicted_class']
        confidence = result['confidence']

        # ─── Result Card ──────────────────────────────────────
        st.markdown(f"""
        <div class="result-card" style="background: {info['color']}22; border: 2px solid {info['color']};">
            <h1 style="color: {info['color']}; margin:0">{info['emoji']}</h1>
            <h2 style="color: {info['color']}; margin:0.5rem 0">{info['label']}</h2>
            <h3 style="color: #555; margin:0">Confidence: {confidence*100:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**📋 Description:** {info['description']}")
        st.markdown(f"**💡 Recommendation:** {info['advice']}")

    # ─── Confidence Chart ─────────────────────────────────────
    st.markdown('<p class="section-header">📊 Confidence Scores</p>', unsafe_allow_html=True)

    confidences = result['all_confidences']
    labels      = list(confidences.keys())
    values      = [v * 100 for v in confidences.values()]
    colors_map  = {c: CLASS_INFO[c]['color'] for c in CLASS_NAMES}
    bar_colors  = [colors_map[l] for l in labels]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in values],
        textposition='outside'
    ))
    fig.update_layout(
        title="Prediction Confidence for Each Stage",
        xaxis_title="Alzheimer's Stage",
        yaxis_title="Confidence (%)",
        yaxis_range=[0, 110],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cccccc"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── All Stages Info ──────────────────────────────────────
    st.markdown('<p class="section-header">ℹ️ Understanding the Stages</p>', unsafe_allow_html=True)

    cols = st.columns(4)
    for i, (class_name, info_item) in enumerate(CLASS_INFO.items()):
        with cols[i]:
            is_predicted = class_name == pred_class
            border = f"3px solid {info_item['color']}" if is_predicted else "1px solid #444"
            st.markdown(f"""
            <div style="border: {border}; border-radius: 10px; padding: 1rem; text-align: center; height: 100%;">
                <h2 style="margin:0">{info_item['emoji']}</h2>
                <p style="color: {info_item['color']}; font-weight: bold; margin: 0.3rem 0; font-size: 0.85rem;">
                    {info_item['label']}
                </p>
                <p style="font-size: 0.75rem; color: #888; margin:0">
                    {info_item['description'][:80]}...
                </p>
                {"<p style='color: " + info_item['color'] + "; font-weight: bold; font-size: 0.8rem;'>← Detected</p>" if is_predicted else ""}
            </div>
            """, unsafe_allow_html=True)

else:
    # ─── Empty State ──────────────────────────────────────────
    st.info("👆 Upload an MRI brain scan image to get started")

    st.markdown('<p class="section-header">ℹ️ How It Works</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Step 1 — Upload**
        Upload a clear MRI brain scan image in JPG or PNG format.
        """)
    with col2:
        st.markdown("""
        **Step 2 — Analyse**
        Our EfficientNetB0 CNN analyses the scan and detects patterns associated with each stage.
        """)
    with col3:
        st.markdown("""
        **Step 3 — Result**
        Get instant classification with confidence scores and medical recommendations.
        """)

    st.markdown('<p class="section-header">📊 About the Model</p>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Architecture", "EfficientNetB0")
    col2.metric("Training Images", "33,984")
    col3.metric("Classes", "4")
    col4.metric("Accuracy", "82%+")
    col5.metric("AUC-ROC", "88%+")

# ─── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    🧠 Alzheimer's Disease Detection | Built with Deep Learning + Streamlit<br>
    Dataset: <a href="https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset" 
    style="color: #6C3483;">Augmented Alzheimer MRI Dataset</a> on Kaggle
</div>
""", unsafe_allow_html=True)