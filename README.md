# Oil Facility Safety Monitor

A deployable, single-page Streamlit web application that combines **two independent AI models** to monitor oil storage facilities in satellite/aerial/drone imagery for both **tank presence** and **fire/smoke hazards** — functioning as an early-warning safety monitoring system.

---

## 🚀 Live Demo
🌐 **[Live Application Link]** *(Replace this after deploying to Streamlit Community Cloud)*

---

## 📸 Application Preview
![App Screenshot](screenshot.png)
*(Run the app locally, take a screenshot, save it as `screenshot.png`)*

---

## 🛢️ Tank Detection Model

### Architecture
A custom Convolutional Neural Network (CNN) trained from scratch:
- **3 Conv2D Blocks:** Conv2D → BatchNormalization → ReLU → MaxPooling → Dropout
- **Classifier Head:** Flatten → Dense → Sigmoid (binary output: tank / no tank)

### Performance
- **Test Accuracy: 99.25%** on patch classification
- Trained on **22,000+ patches** from the **Airbus Oil Storage Detection Dataset** (Kaggle, CC-BY-NC-SA-4.0)

### Data Leakage Prevention
The train/test split was performed on **full satellite images BEFORE patch extraction**. This ensures the test set contains entirely distinct geographic areas, proving the model generalizes to unseen parts of the earth — not just unseen crops of the same image.

### Inference
Sliding window (64×64px, configurable stride) + batch inference + Non-Maximum Suppression (IoU = 0.3).

---

## 🔥 Fire & Smoke Detection Model

### Model
**[`prithivMLmods/Forest-Fire-Detection`](https://huggingface.co/prithivMLmods/Forest-Fire-Detection)** — a fine-tuned **SigLIP2** Vision Transformer loaded directly from Hugging Face.

| Property | Value |
|---|---|
| Architecture | SigLIP2 (Vision Transformer) |
| Classes | `Fire`, `Normal`, `Smoke` |
| Test Accuracy | **99.52%** |
| Source | Hugging Face Hub |
| License | Apache 2.0 |

### Inference
Sliding window (128×128px, configurable stride) across the full uploaded image. The larger window size compared to the tank detector gives the model sufficient visual context to identify fire and smoke patterns. Non-Maximum Suppression is applied independently per class (Fire and Smoke) to prevent cross-class box suppression.

---

## ⚠️ Proximity-Based Safety Alert System

After both models run, the app computes the **minimum boundary distance (in pixels)** between every Fire/Smoke detection and every Tank detection:

| Condition | Alert |
|---|---|
| Fire/smoke within threshold distance of a tank | 🚨 **RED WARNING** — fire/smoke near tank(s), with coordinates |
| Fire/smoke detected but far from tanks | ⚠️ **YELLOW CAUTION** — fire/smoke present, not near tanks |
| No fire/smoke detected | ✅ **GREEN** — no hazards detected |

This logic is inspired by real-world incidents such as drone-strike fires at oil tank farm facilities, where early detection of fire or smoke **near tank clusters** is critical for emergency response decisions.

---

## 🧠 ML Engineering Pattern

This project demonstrates a **realistic production ML pipeline**:
1. **Custom model trained from scratch** on domain-specific data (CNN tank detector)
2. **Pretrained model integration** from a public model hub (SigLIP2 fire/smoke classifier)
3. **Multi-model fusion** into a safety-critical early-warning decision system

---

## 🛠️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/oil-tank-detection-app.git
   cd oil-tank-detection-app
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   The app opens at `http://localhost:8501`. The fire/smoke model (~350MB) downloads automatically on first run and is cached for the session.

---

## 📦 Dataset & Model Credits
- **Tank Dataset:** Airbus Oil Storage Detection Dataset (Kaggle) — License: `CC-BY-NC-SA-4.0`
- **Fire/Smoke Model:** [`prithivMLmods/Forest-Fire-Detection`](https://huggingface.co/prithivMLmods/Forest-Fire-Detection) — License: Apache 2.0

---

## 🧑‍💻 Contributing
Pull requests and issues are welcome! Ideas for improvement:
- Add GPS coordinate overlay if image has EXIF metadata
- Integrate email/SMS alerting on critical detections
- Extend to video frame-by-frame analysis
