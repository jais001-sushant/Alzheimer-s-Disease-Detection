# =============================================================================
# ALZHEIMER'S DETECTION — Phase 1: Full Evaluation
# Author: Sushant Jaiswal | UPES Dehradun
# Run AFTER train.py — loads best saved model
# =============================================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, roc_auc_score
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import shutil
import tempfile
from sklearn.model_selection import train_test_split

print("=" * 65)
print("📊  ALZHEIMER'S DETECTION — Full Evaluation")
print("=" * 65)

# =============================================================================
# CONFIGURATION  — Must match train.py
# =============================================================================

BASE_DATA_DIR = os.getenv(
    "DATASET_PATH",
    "/Users/smilodon002/Downloads/Alzheimer_Dataset/AugmentedAlzheimerDataset"
)

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
CLASS_NAMES = ['MildDemented', 'ModerateDemented',
               'NonDemented', 'VeryMildDemented']

MODEL_PATH  = "best_alzheimer_efficientnet.keras"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# LOAD MODEL
# =============================================================================

print(f"\n   Loading model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"❌  Model not found! Run train.py first.")
    exit(1)

model = keras.models.load_model(MODEL_PATH)
print("   ✅ Model loaded successfully!")

# =============================================================================
# RECREATE TEST GENERATOR
# =============================================================================

print("\n   Recreating test split...")

def create_test_split(data_dir):
    tmp = tempfile.mkdtemp()
    test_dir = os.path.join(tmp, 'test')
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    for cls in CLASS_NAMES:
        cls_path = os.path.join(data_dir, cls)
        imgs = [f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.jfif'))]
        _, rest = train_test_split(imgs, test_size=0.30, random_state=42)
        _, test_imgs = train_test_split(rest, test_size=0.50, random_state=42)
        for img in test_imgs:
            shutil.copy2(os.path.join(cls_path, img),
                         os.path.join(test_dir, cls, img))

    return test_dir, tmp


test_dir, tmp_base = create_test_split(BASE_DATA_DIR)

datagen = ImageDataGenerator(rescale=1. / 255)
test_gen = datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)
print(f"   Test images: {test_gen.samples:,}")

# =============================================================================
# PREDICTIONS
# =============================================================================

print("\n" + "=" * 65)
print("🔮  RUNNING PREDICTIONS ON TEST SET")
print("=" * 65)

test_gen.reset()
predictions = model.predict(test_gen, verbose=1)
pred_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes

# =============================================================================
# METRICS
# =============================================================================

print("\n" + "=" * 65)
print("📋  COMPREHENSIVE METRICS")
print("=" * 65)

# Overall metrics
loss, acc, prec, rec = model.evaluate(test_gen, verbose=0)
f1  = f1_score(true_classes, pred_classes, average='weighted')

y_true_onehot = tf.keras.utils.to_categorical(true_classes, num_classes=4)
auc = roc_auc_score(y_true_onehot, predictions, multi_class='ovr')

print(f"\n   {'Metric':<15} {'Value':>10} {'Percentage':>12}")
print("   " + "-" * 40)
print(f"   {'Accuracy':<15} {acc:>10.4f} {acc*100:>11.2f}%")
print(f"   {'Precision':<15} {prec:>10.4f} {prec*100:>11.2f}%")
print(f"   {'Recall':<15} {rec:>10.4f} {rec*100:>11.2f}%")
print(f"   {'F1-Score':<15} {f1:>10.4f} {f1*100:>11.2f}%")
print(f"   {'AUC-ROC':<15} {auc:>10.4f} {auc*100:>11.2f}%")
print("   " + "-" * 40)

# Per-class report
print("\n   📄 Per-Class Classification Report:")
print(classification_report(true_classes, pred_classes,
                             target_names=CLASS_NAMES, digits=4))

# Save metrics CSV
metrics_df = pd.DataFrame({
    'Metric':     ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Value':      [acc, prec, rec, f1, auc],
    'Percentage': [f'{v*100:.2f}%' for v in [acc, prec, rec, f1, auc]]
})
metrics_df.to_csv(os.path.join(RESULTS_DIR, 'final_metrics.csv'), index=False)
print(f"   💾 Saved: {RESULTS_DIR}/final_metrics.csv")

# =============================================================================
# CONFUSION MATRIX
# =============================================================================

print("\n" + "=" * 65)
print("🔲  CONFUSION MATRIX")
print("=" * 65)

cm = confusion_matrix(true_classes, pred_classes)

# Normalized confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Confusion Matrix — EfficientNetB0 Alzheimer Detection',
             fontsize=15, fontweight='bold', y=1.02)

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
axes[0].set_title('Raw Counts', fontsize=13, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].set_xlabel('Predicted Label', fontsize=11)
axes[0].tick_params(axis='x', rotation=30)
axes[0].tick_params(axis='y', rotation=0)

# Normalized
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
axes[1].set_title('Normalized (%)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].set_xlabel('Predicted Label', fontsize=11)
axes[1].tick_params(axis='x', rotation=30)
axes[1].tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"   📊 Saved: {RESULTS_DIR}/confusion_matrix.png")

# =============================================================================
# METRICS COMPARISON BAR CHART
# =============================================================================

metrics_names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
metrics_values = [acc, prec, rec, f1, auc]
colors = ['#2A9D8F', '#457B9D', '#E9C46A', '#F4A261', '#E63946']

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics_names, [v * 100 for v in metrics_values],
               color=colors, edgecolor='white', width=0.55)
plt.title('Model Performance Metrics — EfficientNetB0',
          fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Score (%)', fontsize=12)
plt.ylim(0, 105)
plt.grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1.0,
             f'{val*100:.2f}%', ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'metrics_bar_chart.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"   📊 Saved: {RESULTS_DIR}/metrics_bar_chart.png")

# =============================================================================
# ROC CURVES (One-vs-Rest per class)
# =============================================================================

from sklearn.metrics import roc_curve, auc as auc_score

plt.figure(figsize=(10, 7))
colors_roc = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A']

for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors_roc)):
    fpr, tpr, _ = roc_curve(y_true_onehot[:, i], predictions[:, i])
    roc_auc_cls  = auc_score(fpr, tpr)
    plt.plot(fpr, tpr, color=color, linewidth=2,
             label=f'{cls_name}  (AUC = {roc_auc_cls:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves — Per Class (One-vs-Rest)',
          fontsize=14, fontweight='bold', pad=15)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"   📊 Saved: {RESULTS_DIR}/roc_curves.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("🎉  EVALUATION COMPLETE!")
print("=" * 65)
print(f"   Test Accuracy  : {acc*100:.2f}%")
print(f"   F1-Score       : {f1*100:.2f}%")
print(f"   AUC-ROC        : {auc*100:.2f}%")
print(f"\n   📁 All results saved in: {RESULTS_DIR}/")
print(f"      ✅ final_metrics.csv")
print(f"      ✅ confusion_matrix.png  (raw + normalized)")
print(f"      ✅ metrics_bar_chart.png")
print(f"      ✅ roc_curves.png")
print(f"\n   ➡️  Next: Run  src/gradcam.py  for explainability heatmaps")
print("=" * 65)

# Cleanup
try:
    shutil.rmtree(tmp_base)
except Exception:
    pass