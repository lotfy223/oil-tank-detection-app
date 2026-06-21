# Oil Storage Tank Detection Web App

A deployable, single-page Streamlit web application that detects oil storage tanks in high-resolution satellite imagery using a pre-trained Convolutional Neural Network (CNN).

The app implements a **sliding-window detector** with **Non-Maximum Suppression (NMS)** to turn a binary patch classifier into an object detector capable of scanning full-sized satellite images.

---

## 🚀 Live Demo
🌐 **[Live Application Link]** *(Placeholder: Replace this link after deploying to Streamlit Community Cloud)*

---

## 📸 Application Preview
![App Screenshot](screenshot.png)
*(Placeholder: Run the app locally, take a screenshot, and save it as `screenshot.png` in this directory)*

---

## 🛠️ Local Installation & Setup

Follow these steps to run the application locally on your machine:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/oil-tank-detection-app.git
   cd oil-tank-detection-app
   ```

2. **Set up a Virtual Environment:**
   *On Windows:*
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
   *On macOS/Linux:*
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
   The application will compile and open in your default web browser (usually at `http://localhost:8501`).

---

## 🔬 Deep Learning & Model Configuration

### 1. CNN Model Architecture
The detector uses a pre-trained Keras CNN model `oil_tank_detection_model.keras` which achieves **99.25% test accuracy** on patch classification.
* **Feature Extractor:** 3 Convolutional blocks, each consisting of `Conv2D` layers, `BatchNormalization` for training stability, `ReLU` activation, `MaxPooling` for dimensionality reduction, and `Dropout` layers to mitigate overfitting.
* **Classifier Head:** A `Flatten` layer transitioning into standard `Dense` (fully-connected) layers, ending in a single-neuron `Sigmoid` classifier outputting a probability between `0.0` (no tank) and `1.0` (tank).

### 2. Preventing Spatial Data Leakage
In geospatial datasets like satellite imagery, random splitting of patches results in extreme **data leakage** because adjacent/overlapping patches from the same image end up in both training and test sets. This leads to an artificially high test score.
* **Prevention:** The train/test split was performed on the **original, full-size satellite images BEFORE patch extraction**. This ensures that the test set consists of entirely distinct geographical areas, confirming that the model generalizes to completely unseen territories.

### 3. Sliding-Window Detector & NMS
Since the CNN is a binary patch classifier trained on `64x64` patches, the app implements a sliding window detection framework:
* **Sliding Window:** Crop a `64x64` window across the image using a configurable **Stride** (step size).
* **Batch Inference:** To optimize speed, all cropped patches are normalized (divided by `255.0`) and passed to the model as a single batch.
* **Non-Maximum Suppression (NMS):** Filters out overlapping bounding boxes pointing to the same tank using an Intersection-over-Union (IoU) threshold of `0.3`.

---

## 📦 Dataset Credits
This model was trained on the **Airbus Oil Storage Detection Dataset** available on Kaggle.
* **License:** Creative Commons `CC-BY-NC-SA-4.0`

---

## 🧑‍💻 Contributing
Feel free to open issues or submit pull requests to enhance the sliding window algorithm, speed up batch inference, or improve the UI styling.
