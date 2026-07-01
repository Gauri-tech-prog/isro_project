import streamlit as st
from PIL import Image
import io
import time
from pipeline import full_pipeline

st.set_page_config(
    page_title="IR Satellite Colorization",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------
# Custom CSS — professional dark theme
# ---------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #131722 0%, #0a0c12 60%);
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2.2rem 1rem 1.4rem 1rem;
        border-bottom: 1px solid #1f2430;
        margin-bottom: 2rem;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00d4ff22, #7b61ff22);
        border: 1px solid #00d4ff44;
        color: #6ee7ff;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 20%, #8fd8ff 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #8b93a7;
        font-size: 1rem;
        margin-top: 0.6rem;
        font-weight: 400;
    }

    /* Section labels */
    .section-label {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, #00d4ff14, #7b61ff14);
        border: 1px solid #00d4ff33;
        padding: 0.45rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #6ee7ff;
        margin-bottom: 0.8rem;
    }

    /* Cards */
    .card {
        background: #151925;
        border: 1px solid #232838;
        border-radius: 14px;
        padding: 1.3rem;
        transition: border-color 0.2s ease;
    }
    .card:hover { border-color: #00d4ff55; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1017;
        border-right: 1px solid #1f2430;
    }
    section[data-testid="stSidebar"] .stMetric {
        background: #151925;
        border: 1px solid #232838;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.6rem;
    }
    section[data-testid="stSidebar"] h3 {
        color: #6ee7ff;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1.5rem;
    }

    /* Upload dropzone */
    .upload-empty {
        text-align: center;
        padding: 3.5rem 2rem;
        color: #5b6478;
        border: 1.5px dashed #2a3040;
        border-radius: 16px;
        background: #10131c;
    }
    .upload-empty h3 { color: #cfd6e4; font-weight: 600; margin-bottom: 0.4rem; }

    /* Buttons */
    .stDownloadButton button {
        background: linear-gradient(135deg, #00d4ff, #7b61ff) !important;
        color: #05070d !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1rem !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #10131c;
        border: 1px dashed #2a3040;
        border-radius: 14px;
        padding: 0.5rem;
    }

    /* Success banner */
    .stAlert { border-radius: 10px; }

    hr { border-color: #1f2430 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-badge">🛰️ Pix2Pix GAN · Landsat 8/9</div>
    <div class="hero-title">IR Satellite Image Colorization</div>
    <div class="hero-subtitle">CLAHE Sharpening → Super Resolution → GAN Colorization</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Model Metrics")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("PSNR", "13.96 dB")
    with c2:
        st.metric("SSIM", "0.3205")
    st.metric("Inference Speed", "0.5 ms / tile")

    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    <div class="card" style="font-size:0.9rem; line-height:1.9;">
    <b>Generator</b> — U-Net<br>
    <b>Discriminator</b> — PatchGAN<br>
    <b>Loss</b> — MSE + L1 (λ=100)<br>
    <b>Dataset</b> — Landsat 8/9<br>
    <b>Tiles</b> — 1,594 paired<br>
    <b>Region</b> — Mumbai, India
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎨 Color Mapping")
    st.markdown("""
    <div class="card" style="font-size:0.9rem; line-height:1.9;">
    🟢 Vegetation<br>
    🔵 Water<br>
    ⬜ Urban / Built-up<br>
    🟤 Bare Soil
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# Upload
# ---------------------------------------------------------------
uploaded = st.file_uploader(
    "📥 Upload NIR / Infrared Satellite Image",
    type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
    help="Upload a grayscale NIR satellite tile"
)

if uploaded is None:
    st.markdown("""
    <div class="upload-empty">
        <h3>⬆️ Upload a NIR Satellite Image to Begin</h3>
        <p>Supports PNG, JPG, JPEG, TIF</p>
    </div>
    """, unsafe_allow_html=True)

else:
    input_img = Image.open(uploaded)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">📥 Original NIR Input</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(input_img.convert('L'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("🔄 Running pipeline..."):
        t0 = time.time()
        sharpened, sr_img, rgb_out = full_pipeline(input_img)
        elapsed = (time.time() - t0) * 1000

    with col2:
        st.markdown('<div class="section-label">🎨 Colorized RGB Output</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(rgb_out, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.success(f"✅ Processing complete in {elapsed:.0f} ms")

    st.markdown("---")
    st.markdown("### 🔬 Pipeline Steps")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="section-label">Step 1 · CLAHE + Sharpening</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(sharpened, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-label">Step 2 · Super Resolution 2×</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(sr_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="section-label">Step 3 · RGB Colorization</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(rgb_out, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    buf = io.BytesIO()
    rgb_out.save(buf, format='PNG')
    st.download_button(
        label="⬇️ Download Colorized RGB Image",
        data=buf.getvalue(),
        file_name="colorized_satellite.png",
        mime="image/png",
        use_container_width=True
    )