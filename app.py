import streamlit as st
from PIL import Image, ImageDraw
import pandas as pd
import time
import os

from model_utils import load_detection_model
from detector import sliding_window_detection

# Set page config for a premium look and feel
st.set_page_config(
    page_title="Oil Storage Tank Detector",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling to enhance UI aesthetics
st.markdown("""
    <style>
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #64748B;
        font-size: 1.15rem;
        margin-top: 5px;
        margin-bottom: 25px;
    }
    .metric-container {
        background-color: #F8FAFC;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        text-align: center;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #F0FDF4;
        border: 1px solid #DCFCE7;
        color: #166534;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<h1 class="main-title">🛢️ Oil Storage Tank Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">CNN-based detector trained on Airbus satellite data (99.25% test accuracy on patch classification)</p>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("🔧 Detector Configuration")
st.sidebar.markdown("Fine-tune the sliding-window detector parameters below:")

# Sliders
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.3,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Minimum model confidence score (probability) required to classify a window as an oil storage tank."
)

stride = st.sidebar.slider(
    "Stride (px)",
    min_value=16,
    max_value=64,
    value=32,
    step=4,
    help="Step size for sliding the detection window. Smaller values (e.g., 16px) increase overlap and accuracy but slow down processing."
)

st.sidebar.info(
    "💡 **Note:** Stride controls the window overlap. A smaller stride increases resolution but generates more patches, causing detection to take slightly longer."
)

# Load the model
try:
    with st.spinner("Initializing Deep Learning Engine..."):
        model, backend = load_detection_model()
    st.sidebar.success(f"🤖 Keras active backend: **{backend.upper()}**")
except Exception as e:
    st.error(f"Failed to load CNN model: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader(
    "Upload a satellite image (JPEG, JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a full-sized satellite image to scan for oil storage tanks."
)

if uploaded_file is not None:
    # 1. Load and display original image
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Error opening image file: {e}")
        st.stop()

    # Create layout columns for a dashboard feel
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Analysis View")
        
    with col2:
        st.subheader("Detection Metrics")

    # 2. Run detection
    t_start = time.time()
    with st.spinner("Scanning satellite image using sliding-window detector..."):
        try:
            boxes = sliding_window_detection(
                image=image,
                model=model,
                confidence_threshold=confidence_threshold,
                stride=stride,
                iou_threshold=0.3
            )
            detection_time = time.time() - t_start
        except Exception as e:
            st.error(f"An error occurred during detection processing: {e}")
            st.stop()

    # 3. Present Results
    with col2:
        # Display detection summary metrics
        st.markdown(
            f"""
            <div class="metric-container">
                <span style="font-size: 14px; color: #64748B; font-weight: 500;">Tanks Detected</span><br>
                <span style="font-size: 36px; font-weight: 800; color: #FF3B30;">{len(boxes)}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Performance metric
        st.markdown(
            f"<p style='color: #64748B; font-size: 0.85rem; text-align: center; margin-top: -10px;'>"
            f"Scanned in {detection_time:.2f} seconds</p>", 
            unsafe_allow_html=True
        )

        # Raw detections list
        if boxes:
            with st.expander("📋 Raw Detections List", expanded=False):
                # Format boxes as a DataFrame for pretty table view
                df = pd.DataFrame(
                    boxes, 
                    columns=["x1", "y1", "x2", "y2", "Confidence"]
                )
                # Re-order and style columns
                df["Confidence"] = df["Confidence"].apply(lambda val: f"{val:.4f}")
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No tanks detected with current settings. Try lowering the Confidence Threshold.")

    with col1:
        # Render image with bounding boxes
        if boxes:
            image_with_boxes = image.copy()
            draw = ImageDraw.Draw(image_with_boxes)
            
            # Dynamic line width based on image dimensions
            line_width = max(2, int(min(image.size) / 200))
            
            for box in boxes:
                x1, y1, x2, y2, conf = box
                draw.rectangle([x1, y1, x2, y2], outline="#FF3B30", width=line_width)
            
            st.image(image_with_boxes, use_container_width=True, caption="Detected Oil Storage Tanks (Red Boxes)")
        else:
            # Display clean original image with notification
            st.image(image, use_container_width=True, caption="Uploaded Satellite Image")
            st.warning("⚠️ No oil storage tanks were detected in the uploaded image. If you believe this is incorrect, try reducing the Confidence Threshold in the sidebar.")

else:
    # Upload instructions / landing state
    st.info("👆 Please upload a satellite image using the file uploader above to begin scanning.")

# How it works section (always at bottom)
st.markdown("---")
with st.expander("🔬 How it Works & Model Details", expanded=False):
    st.markdown("""
    ### Deep Learning Oil Tank Detector
    
    This application utilizes a pre-trained **Convolutional Neural Network (CNN)** to identify oil storage tanks in high-resolution satellite imagery. Because the model is a binary patch classifier (detecting whether a 64x64 pixel patch contains a tank or not), the app implements a custom sliding-window mechanism to perform object detection over arbitrary image sizes.
    
    #### 1. Sliding Window & Batch Inference
    - **Sliding Window:** The app slides a **64x64 pixel window** across the uploaded image using a configurable step size (**Stride**).
    - **Normalization:** Each patch is resized and normalized (pixel values divided by `255.0`) to match the exact training distribution.
    - **Batch Processing:** To maximize CPU/GPU execution speed, all patches are grouped into a single batch and run through the model at once, rather than predicting patch-by-patch in a slow loop.
    - **Non-Maximum Suppression (NMS):** Sliding windows with overlaps often flag the same tank multiple times. The detector applies NMS with an **Intersection-over-Union (IoU) threshold of 0.3** to merge overlapping predictions into a single, clean bounding box.
    
    #### 2. CNN Architecture
    The model is structured with:
    - **3 Conv2D Blocks:** Each containing a 2D Convolutional layer, Batch Normalization (to stabilize training), ReLU activation, MaxPooling (to reduce spatial dimensionality), and Dropout (to prevent overfitting).
    - **Classifier Head:** A Flatten layer followed by Dense (fully connected) layers, ending in a single-neuron Sigmoid output which outputs a probability between `0.0` (no tank) and `1.0` (tank).
    
    #### 3. Preventing Data Leakage
    - **The Problem:** Satellite imagery patches taken from the same image are highly correlated. Splitting patches randomly into training and testing sets causes **data leakage** because patches from the same location end up in both sets, leading to artificially inflated test accuracy.
    - **The Solution:** The dataset split (Train/Test) was performed on the **original, full-size satellite images BEFORE patch extraction**. This guarantees the test set consists of entirely distinct geographical areas, proving the model generalizes to completely unseen parts of the earth.
    
    #### 4. Dataset Credit
    - Trained on the **Airbus Oil Storage Detection Dataset** (Kaggle), licensed under `CC-BY-NC-SA-4.0`.
    - Achieves **99.25% test accuracy** on patch classification.
    """)
