# 🧠 Alzheimer's Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)

> AI-powered Alzheimer's disease detection from brain MRI scans using **EfficientNetB0 Transfer Learning** with **Grad-CAM explainability**

---

## 🎯 Live Demo

**🔗 [Try the Web App →](https://your-app.streamlit.app)**

Upload any brain MRI scan and get an instant prediction with visual explainability!

---

## 📊 Model Performance

| Metric     | Score     |
|------------|-----------|
| Accuracy   | **~90%**  |
| Precision  | **~91%**  |
| Recall     | **~89%**  |
| F1-Score   | **~90%**  |
| AUC-ROC    | **~96%**  |

> Trained on 33,984 MRI images. Baseline custom CNN achieved 77.13%. EfficientNetB0 transfer learning improved this significantly.

---

## 🖥️ Screenshots

| Upload & Predict | Grad-CAM Heatmap |
|:-:|:-:|
| ![App Screenshot](results/app_screenshot.png) | ![Gradcam](results/gradcam/gradcam_MildDemented.png) |

---

## 🏗️ Architecture

```
Input MRI (224×224×3)
        ↓
EfficientNetB0 (pretrained ImageNet)
  └─ Stage 1: Frozen base → train custom head
  └─ Stage 2: Unfreeze top 30 layers → fine-tune
        ↓
GlobalAveragePooling2D
        ↓
Dense(256) → Dropout(0.4)
        ↓
Dense(128) → Dropout(0.3)
        ↓
Dense(4, softmax)  [4 Alzheimer stages]
```

### 🔥 Why EfficientNetB0?
- **5.3M params** vs 17M in custom CNN — lighter & faster
- **Pretrained on ImageNet** — already knows edges, textures, shapes
- **Best accuracy/compute tradeoff** for M2 Mac training
- **~25MB model size** — no GitHub file size issues

---

## 🧬 Dataset

**Augmented Alzheimer MRI Dataset** — [Kaggle Link](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)

| Class | Images | Description |
|---|---|---|
| NonDemented | 9,600 | No cognitive impairment |
| VeryMildDemented | 8,960 | Early stage cognitive decline |
| MildDemented | 8,960 | Moderate cognitive impairment |
| ModerateDemented | 6,464 | Advanced cognitive impairment |
| **Total** | **33,984** | |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/jais001-sushant/alzheimer-detection.git
cd alzheimer-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
Download from [Kaggle](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset) and set the path:
```bash
export DATASET_PATH="/path/to/AugmentedAlzheimerDataset"
```

### 4. Train the model
```bash
python src/train.py
```
> ⏱️ Expected time on M2 Mac: ~1.5-2 hours (Stage 1 + Stage 2)

### 5. Evaluate
```bash
python src/evaluate.py
```

### 6. Generate Grad-CAM heatmaps
```bash
python src/gradcam.py
```

### 7. Launch web app
```bash
streamlit run app/app.py
```

---

## 📁 Project Structure

```
alzheimer-detection/
│
├── src/
│   ├── train.py          # EfficientNetB0 2-stage training
│   ├── evaluate.py       # Full metrics + confusion matrix + ROC
│   └── gradcam.py        # Grad-CAM explainability heatmaps
│
├── app/
│   └── app.py            # Streamlit web app (dark medical theme)
│
├── results/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── metrics_bar_chart.png
│   ├── class_distribution.png
│   └── gradcam/          # Per-class Grad-CAM heatmaps
│
├── backup/
│   └── alzheimer_detection.py   # Original CNN (v1)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔥 Key Features

- ✅ **Transfer Learning** — EfficientNetB0 pretrained on ImageNet
- ✅ **2-Stage Training** — Frozen base → fine-tune top layers
- ✅ **Grad-CAM** — Visual explainability of predictions
- ✅ **Class Balancing** — Weighted loss for imbalanced classes
- ✅ **MRI-specific Augmentation** — Rotation, zoom, brightness
- ✅ **Live Streamlit App** — Upload MRI → instant prediction
- ✅ **Full Metrics** — Accuracy, Precision, Recall, F1, AUC-ROC, ROC curves
- ✅ **Dark Medical UI** — Professional hospital dashboard theme

---

## 👨‍💻 Author

**Sushant Jaiswal**
- 🎓 B.Tech CSE (AI/ML) — UPES Dehradun, 3rd Year
- 💼 [LinkedIn](https://linkedin.com/in/sushant-jaiswal-ba5996298)
- 🐙 [GitHub](https://github.com/jais001-sushant)
- 📧 jaiswal001sushant@gmail.com

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## ⚕️ Disclaimer

This tool is for **research and educational purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified neurologist or healthcare professional.

---

⭐ If this project helped you, please give it a star!