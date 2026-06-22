import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import time
import os

from model_utils import load_detection_model
from detector import sliding_window_detection
from fire_detector import load_fire_model, sliding_window_fire_detection

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oil Facility Safety Monitor",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-title {
        font-weight: 800; font-size: 2rem; color: #1E293B;
        margin-bottom: 0px; padding-bottom: 0px;
    }
    .subtitle {
        color: #64748B; font-size: 1.05rem;
        margin-top: 4px; margin-bottom: 20px;
    }
    .metric-card {
        background: #F8FAFC; padding: 16px 10px;
        border-radius: 14px; border: 1px solid #E2E8F0;
        text-align: center; margin-bottom: 12px;
    }
    .metric-label { font-size: 13px; color: #64748B; font-weight: 500; }
    .metric-value { font-size: 38px; font-weight: 800; line-height: 1.1; }
    .alert-red {
        background: #FFF1F1; border: 2px solid #FF3B30;
        border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;
        color: #B91C1C; font-weight: 600; font-size: 1.05rem;
    }
    .alert-yellow {
        background: #FFFBEB; border: 2px solid #F59E0B;
        border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;
        color: #92400E; font-weight: 600; font-size: 1.05rem;
    }
    .alert-green {
        background: #F0FDF4; border: 2px solid #22C55E;
        border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;
        color: #166534; font-weight: 600; font-size: 1.05rem;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🛢️ Oil Facility Safety Monitor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Dual-model AI pipeline — Tank detection (custom CNN, 99.25% accuracy) '
    '+ Fire/Smoke detection (SigLIP2, 99.52% accuracy) for early-warning safety monitoring.</p>',
    unsafe_allow_html=True
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Detector Configuration")

st.sidebar.subheader("🛢️ Tank Detection")
confidence_threshold = st.sidebar.slider(
    "Tank Confidence Threshold", 0.3, 0.9, 0.5, 0.05,
    help="Minimum probability to classify a window as a tank."
)
stride = st.sidebar.slider(
    "Tank Stride (px)", 16, 64, 32, 4,
    help="Smaller stride = more accurate but slower."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔥 Fire & Smoke Detection")
enable_fire = st.sidebar.checkbox("Enable Fire/Smoke Detection", value=True)

fire_confidence = st.sidebar.slider(
    "Fire/Smoke Confidence Threshold", 0.3, 0.9, 0.5, 0.05,
    help="Minimum probability to classify a window as Fire or Smoke.",
    disabled=not enable_fire
)
fire_stride = st.sidebar.slider(
    "Fire/Smoke Stride (px)", 32, 128, 64, 16,
    help="Step size for the 128x128 fire/smoke window.",
    disabled=not enable_fire
)
proximity_px = st.sidebar.slider(
    "⚠️ Proximity Alarm Distance (px)", 0, 200, 50, 10,
    help="If a fire/smoke box is within this distance of a tank box, trigger a red alert.",
    disabled=not enable_fire
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Smaller strides increase detection resolution but take longer to process.")

# ─── Model Loading ───────────────────────────────────────────────────────────
try:
    with st.spinner("🤖 Initializing Tank Detection Engine..."):
        tank_model, backend = load_detection_model()
    st.sidebar.success(f"🛢️ Tank model ready · Backend: **{backend.upper()}**")
except Exception as e:
    st.error(f"Failed to load tank model: {e}")
    st.stop()

if enable_fire:
    try:
        with st.spinner("🔥 Loading Fire/Smoke Detection Model (first run downloads ~350MB)..."):
            fire_model, fire_processor = load_fire_model()
        st.sidebar.success("🔥 Fire/Smoke model ready")
    except Exception as e:
        st.error(f"Failed to load fire/smoke model: {e}")
        st.stop()

# ─── File Uploader ───────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a satellite / aerial / drone image (JPEG, JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a full-resolution image of an oil storage facility."
)


def box_distance(boxA, boxB):
    """Minimum boundary distance between two boxes (0 if overlapping)."""
    ax1, ay1, ax2, ay2 = boxA[:4]
    bx1, by1, bx2, by2 = boxB[:4]
    dx = max(0.0, ax1 - bx2, bx1 - ax2)
    dy = max(0.0, ay1 - by2, by1 - ay2)
    return (dx ** 2 + dy ** 2) ** 0.5


def draw_label(draw, text, x1, y1, color, font_size=13):
    """Draw a small filled label tag above a bounding box."""
    # Build a text box background
    text_bbox = draw.textbbox((x1, y1), text)
    text_w = text_bbox[2] - text_bbox[0] + 6
    text_h = text_bbox[3] - text_bbox[1] + 4
    tag_y = max(0, y1 - text_h - 2)
    draw.rectangle([x1, tag_y, x1 + text_w, tag_y + text_h], fill=color)
    draw.text((x1 + 3, tag_y + 2), text, fill="white")


# ─── Main Processing ─────────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error opening image: {e}")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📡 Analysis View")
    with col2:
        st.subheader("📊 Detection Metrics")

    # ── Tank Detection ───────────────────────────────────────────────────────
    t_start = time.time()
    with st.spinner("🔍 Scanning for oil storage tanks..."):
        try:
            tank_boxes = sliding_window_detection(
                image=image, model=tank_model,
                confidence_threshold=confidence_threshold,
                stride=stride, iou_threshold=0.3
            )
        except Exception as e:
            st.error(f"Tank detection error: {e}")
            st.stop()

    # ── Fire / Smoke Detection ───────────────────────────────────────────────
    fire_boxes = []
    if enable_fire:
        with st.spinner("🔥 Scanning for fire and smoke..."):
            try:
                fire_boxes = sliding_window_fire_detection(
                    image=image, model=fire_model, processor=fire_processor,
                    confidence_threshold=fire_confidence,
                    win_size=128, stride=fire_stride, iou_threshold=0.3
                )
            except Exception as e:
                st.error(f"Fire/smoke detection error: {e}")
                fire_boxes = []

    total_time = time.time() - t_start

    fire_only = [b for b in fire_boxes if b[4] == "Fire"]
    smoke_only = [b for b in fire_boxes if b[4] == "Smoke"]

    # ── Proximity Alert Logic ─────────────────────────────────────────────────
    alert_html = ""
    affected_tanks = []
    if enable_fire:
        if fire_boxes:
            # Check if any fire/smoke box is within proximity_px of any tank box
            near_tank = False
            for fb in fire_boxes:
                for tb in tank_boxes:
                    if box_distance(fb, tb) <= proximity_px:
                        near_tank = True
                        coord_str = f"({int(tb[0])},{int(tb[1])})-({int(tb[2])},{int(tb[3])})"
                        if coord_str not in affected_tanks:
                            affected_tanks.append(coord_str)

            if near_tank and tank_boxes:
                tanks_str = ", ".join(affected_tanks)
                alert_html = (
                    f'<div class="alert-red">🚨 <b>WARNING:</b> Fire/smoke detected near '
                    f'oil storage tank(s)!<br>'
                    f'<small>Affected tank(s): {tanks_str}</small></div>'
                )
            else:
                alert_html = (
                    '<div class="alert-yellow">⚠️ Fire/smoke detected in image, '
                    'but <b>not</b> near a detected tank.</div>'
                )
        else:
            alert_html = '<div class="alert-green">✅ No fire or smoke detected.</div>'

    # ── Metrics Column ────────────────────────────────────────────────────────
    with col2:
        if enable_fire:
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">🛢️ Tanks</div>'
                    f'<div class="metric-value" style="color:#FF3B30">{len(tank_boxes)}</div></div>',
                    unsafe_allow_html=True
                )
            with mcol2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">🔥 Fire Zones</div>'
                    f'<div class="metric-value" style="color:#FF9500">{len(fire_only)}</div></div>',
                    unsafe_allow_html=True
                )
            with mcol3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">💨 Smoke Zones</div>'
                    f'<div class="metric-value" style="color:#6B7280">{len(smoke_only)}</div></div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">🛢️ Tanks Detected</div>'
                f'<div class="metric-value" style="color:#FF3B30">{len(tank_boxes)}</div></div>',
                unsafe_allow_html=True
            )

        st.markdown(
            f"<p style='color:#94A3B8;font-size:0.82rem;text-align:center;margin-top:-6px;'>"
            f"Processed in {total_time:.2f}s</p>", unsafe_allow_html=True
        )

        # ── Safety Alert ──────────────────────────────────────────────────────
        if alert_html:
            st.markdown(alert_html, unsafe_allow_html=True)

        # ── Raw Detections Table ──────────────────────────────────────────────
        all_raw = []
        for b in tank_boxes:
            all_raw.append({"Type": "🛢️ Tank", "x1": int(b[0]), "y1": int(b[1]),
                             "x2": int(b[2]), "y2": int(b[3]), "Conf": f"{b[4]:.3f}"})
        for b in fire_boxes:
            emoji = "🔥" if b[4] == "Fire" else "💨"
            all_raw.append({"Type": f"{emoji} {b[4]}", "x1": int(b[0]), "y1": int(b[1]),
                             "x2": int(b[2]), "y2": int(b[3]), "Conf": f"{b[5]:.3f}"})
        if all_raw:
            with st.expander("📋 Raw Detections", expanded=False):
                st.dataframe(pd.DataFrame(all_raw), use_container_width=True)
        else:
            st.info("No detections with current settings.")

    # ── Render Combined Image ─────────────────────────────────────────────────
    with col1:
        result_img = image.copy()
        draw = ImageDraw.Draw(result_img)
        lw = max(2, int(min(image.size) / 200))

        # Tanks → Red
        for box in tank_boxes:
            x1, y1, x2, y2, conf = box
            draw.rectangle([x1, y1, x2, y2], outline="#FF3B30", width=lw)

        # Fire → Orange | Smoke → Gray
        color_map = {"Fire": "#FF9500", "Smoke": "#6B7280"}
        emoji_map = {"Fire": "🔥 Fire", "Smoke": "💨 Smoke"}
        for box in fire_boxes:
            x1, y1, x2, y2, label, conf = box
            color = color_map[label]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
            draw_label(draw, emoji_map[label], x1, y1, color)

        caption_parts = []
        if tank_boxes:
            caption_parts.append(f"🛢️ {len(tank_boxes)} Tank(s) — Red")
        if fire_only:
            caption_parts.append(f"🔥 {len(fire_only)} Fire Zone(s) — Orange")
        if smoke_only:
            caption_parts.append(f"💨 {len(smoke_only)} Smoke Zone(s) — Gray")
        caption = "  |  ".join(caption_parts) if caption_parts else "No detections"

        st.image(result_img, use_container_width=True, caption=caption)

        if not tank_boxes and not fire_boxes:
            st.warning("No detections found. Try lowering the confidence thresholds in the sidebar.")

else:
    st.info("👆 Upload a satellite or aerial image above to begin safety analysis.")

# ─── How It Works ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔬 How It Works — Dual-Model Pipeline", expanded=False):
    st.markdown("""
    ### Two-Model AI Safety Monitoring System

    This application combines **two independent deep-learning models** to deliver a real-time
    early-warning system for oil storage facility safety monitoring.

    ---

    #### 🛢️ Model 1 — Custom CNN Tank Detector
    - **Architecture:** 3 Conv2D blocks (Conv2D → BatchNorm → ReLU → MaxPool → Dropout) + Dense classifier with Sigmoid output.
    - **Training Data:** 22,000+ patches extracted from the **Airbus Oil Storage Detection Dataset** (Kaggle, CC-BY-NC-SA-4.0).
    - **Test Accuracy:** **99.25%** on patch classification.
    - **Inference method:** Sliding window (64×64px, configurable stride) with batch inference and Non-Maximum Suppression (IoU = 0.3).
    - **Data leakage prevention:** Train/test split was performed on **full satellite images** before patch extraction — completely separate geographic areas in test set.

    ---

    #### 🔥 Model 2 — Pretrained SigLIP2 Fire/Smoke Classifier
    - **Model:** [`prithivMLmods/Forest-Fire-Detection`](https://huggingface.co/prithivMLmods/Forest-Fire-Detection) — a fine-tuned SigLIP2 vision transformer.
    - **Classes:** `Fire`, `Normal`, `Smoke` (3-class softmax output).
    - **Test Accuracy:** **99.52%** on its own benchmark test set.
    - **Inference method:** Sliding window (128×128px, larger windows to capture visual context) with per-class NMS independently applied to Fire and Smoke candidates.
    - **Integration pattern:** This is an **off-the-shelf pretrained model** loaded directly from Hugging Face — no training required, demonstrating production-level model integration.

    ---

    #### ⚠️ Proximity-Based Safety Alert System
    - After both detectors run, the app computes the **minimum boundary distance** (in pixels) between every Fire/Smoke box and every Tank box.
    - If any fire/smoke detection is within the configured **Proximity Alarm Distance** of a tank, a **🚨 RED critical alert** is raised with the affected tank coordinates.
    - If fire/smoke is present but far from tanks, a **⚠️ YELLOW caution** is shown.
    - If no fire/smoke is detected, a **✅ GREEN confirmation** is displayed.
    - This logic is inspired by real-world incidents such as drone-strike fires at oil storage facilities, where early detection of fire/smoke near tank clusters is critical for emergency response.

    ---

    #### 🧠 ML Engineering Pattern Demonstrated
    This project showcases a **realistic production ML pipeline**:
    1. Training a model from scratch on domain-specific data (custom CNN).
    2. Integrating a pretrained large model from a public repository (HuggingFace SigLIP2).
    3. Combining both into a safety-critical decision system with domain-aware alert logic.
    """)
