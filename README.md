# 🧠 Alzheimer's Disease Detection

An AI-powered deep learning system that analyses MRI brain scans to detect and classify Alzheimer's disease stages using EfficientNetB0 transfer learning.

---

## 🖼️ Preview
![App Preview](results/confusion_matrix.png)

---

## 🚀 Live Demo
👉 [Click here to try the app](https://your-app-link.streamlit.app)

---

## ✨ Features

- 📤 **MRI Image Upload** — Upload brain scan images for instant analysis
- 🧠 **4-Stage Classification** — NonDemented, VeryMild, Mild, ModerateDemented
- 📊 **Confidence Scores** — Visual bar chart for all 4 predictions
- 💡 **Medical Explanations** — Plain English description of each result
- ⚠️ **Medical Disclaimer** — Clear guidance for professional consultation
- 🎨 **Color Coded Results** — Green/Yellow/Orange/Red severity indicators

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **Architecture** | EfficientNetB0 (Transfer Learning) |
| **Training Images** | 33,984 MRI scans |
| **Accuracy** | 82%+ |
| **AUC-ROC** | 88%+ |
| **Classes** | 4 |

---

## 🔬 Detection Classes

| Class | Description |
|---|---|
| 🟢 NonDemented | No signs of cognitive impairment |
| 🟡 VeryMildDemented | Very early stage — subtle changes |
| 🟠 MildDemented | Moderate cognitive impairment |
| 🔴 ModerateDemented | Advanced stage — significant impairment |

---

## 🛠️ Tech Stack

- **Python** — Core language
- **TensorFlow / Keras** — Deep learning framework
- **EfficientNetB0** — Transfer learning backbone
- **Streamlit** — Web interface
- **Plotly** — Interactive charts
- **Scikit-learn** — Evaluation metrics
- **OpenCV / Pillow** — Image processing

---

## 📂 Project Structure

```
Alzheimer-Disease-Detection/
│
├── app.py                  ← Streamlit web app
├── src/
│   ├── train.py            ← Training script (EfficientNetB0)
│   └── predict.py          ← Prediction functions
├── model/
│   └── alzheimer_model.h5  ← Trained model (see setup below)
├── results/
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── class_distribution.png
│   └── model_results.csv
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🗃️ Dataset

**Augmented Alzheimer MRI Dataset** from Kaggle:
👉 [https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)

```
NonDemented      → 9,600 images
VeryMildDemented → 8,960 images
MildDemented     → 8,960 images
ModerateDemented → 6,464 images
Total            → 33,984 images
```

---

## 📦 Installation (Local)

**Step 1 — Clone the repo**
```bash
git clone https://github.com/jais001-sushant/Alzheimer-Disease-Detection.git
cd Alzheimer-Disease-Detection
```

**Step 2 — Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Download the trained model**

Download `alzheimer_model.h5` from Google Drive and place it in the `model/` folder:
👉 [Download Model](https://drive.google.com/your-model-link-here)

**Step 5 — Run the app**
```bash
streamlit run app.py
```

---

## 🔁 Retrain the Model

If you want to retrain with your own dataset:

```bash
# Set dataset path
export DATASET_PATH="/path/to/AugmentedAlzheimerDataset"

# Run training
python src/train.py
```

Training uses **2-phase transfer learning:**
- Phase 1 → Train classifier head (base frozen)
- Phase 2 → Fine-tune top 30 layers of EfficientNetB0

---

## ⚠️ Medical Disclaimer

This tool is built for **educational and research purposes only**. It is **NOT** a substitute for professional medical diagnosis. Always consult a qualified neurologist or healthcare professional for medical advice and diagnosis.

---

## 📌 Status
> ✅ Completed — Deep Learning Project