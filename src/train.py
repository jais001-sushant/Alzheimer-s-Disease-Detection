import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint, CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import tempfile
from sklearn.model_selection import train_test_split

print("=" * 65)
print("🧠  ALZHEIMER'S DETECTION — EfficientNetB0 Transfer Learning")
print("=" * 65)
print(f"✅  TensorFlow  : {tf.__version__}")
print(f"✅  Keras       : {keras.__version__}")

BASE_DATA_DIR = os.getenv(
    "DATASET_PATH",
    "/Users/smilodon002/Downloads/Alzheimer_Dataset/AugmentedAlzheimerDataset"
)

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS_FROZEN = 15
EPOCHS_FINE   = 25
LR_FROZEN     = 1e-3
LR_FINE       = 1e-5

CLASS_NAMES   = ['MildDemented', 'ModerateDemented',
                 'NonDemented', 'VeryMildDemented']

MODEL_SAVE_PATH = "best_alzheimer_efficientnet.keras"
RESULTS_DIR     = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"\n⚙️  Config loaded:")
print(f"   Dataset  : {BASE_DATA_DIR}")
print(f"   IMG Size : {IMG_SIZE}")
print(f"   Batch    : {BATCH_SIZE}")


def explore_dataset(data_dir):
    print("\n" + "=" * 65)
    print("📊  STEP 1 — DATASET EXPLORATION")
    print("=" * 65)

    if not os.path.exists(data_dir):
        print(f"❌  Dataset not found at: {data_dir}")
        print("    Set env var DATASET_PATH or edit BASE_DATA_DIR above.")
        return False, {}

    total = 0
    class_counts = {}
    for cls in CLASS_NAMES:
        cls_path = os.path.join(data_dir, cls)
        if os.path.exists(cls_path):
            imgs = [f for f in os.listdir(cls_path)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg', '.jfif'))]
            class_counts[cls] = len(imgs)
            total += len(imgs)
            print(f"   {cls:<22}: {len(imgs):>6,} images")
        else:
            print(f"   ❌ {cls} — directory missing")
            class_counts[cls] = 0

    print(f"\n   TOTAL IMAGES : {total:,}")

    # Plot class distribution
    colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A']
    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color=colors, edgecolor='white')
    plt.title('Class Distribution — Alzheimer MRI Dataset', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Alzheimer Stage', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=20, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, (cls, cnt) in zip(bars, class_counts.items()):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 80,
                 f'{cnt:,}', ha='center', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'class_distribution.png'), dpi=150)
    plt.close()
    print(f"\n   📊 Saved: {RESULTS_DIR}/class_distribution.png")
    return True, class_counts


success, class_counts = explore_dataset(BASE_DATA_DIR)
if not success:
    exit(1)


def create_split(data_dir):
    print("\n" + "=" * 65)
    print("🔀  STEP 2 — CREATING 70 / 15 / 15 SPLIT")
    print("=" * 65)

    tmp = tempfile.mkdtemp()
    splits = {}
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(tmp, split)
        splits[split] = split_dir
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

    for cls in CLASS_NAMES:
        cls_path = os.path.join(data_dir, cls)
        imgs = [f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.jfif'))]

        train_imgs, rest = train_test_split(imgs, test_size=0.30, random_state=42)
        val_imgs, test_imgs = train_test_split(rest, test_size=0.50, random_state=42)

        for img in train_imgs:
            shutil.copy2(os.path.join(cls_path, img),
                         os.path.join(splits['train'], cls, img))
        for img in val_imgs:
            shutil.copy2(os.path.join(cls_path, img),
                         os.path.join(splits['val'], cls, img))
        for img in test_imgs:
            shutil.copy2(os.path.join(cls_path, img),
                         os.path.join(splits['test'], cls, img))

        print(f"   {cls:<22}: {len(train_imgs):>5} train | "
              f"{len(val_imgs):>4} val | {len(test_imgs):>4} test")

    return splits['train'], splits['val'], splits['test'], tmp


TRAIN_DIR, VAL_DIR, TEST_DIR, TMP_BASE = create_split(BASE_DATA_DIR)
print("\n   ✅ Split created!")


print("\n" + "=" * 65)
print("🔄  STEP 3 — DATA GENERATORS & AUGMENTATION")
print("=" * 65)


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    zoom_range=0.10,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True, seed=42
)
val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)
test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

print(f"\n   Train  : {train_gen.samples:,} images")
print(f"   Val    : {val_gen.samples:,} images")
print(f"   Test   : {test_gen.samples:,} images")

# Class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight_dict = dict(enumerate(class_weights))
print("\n   ⚖️  Class weights:")
for i, cls in enumerate(CLASS_NAMES):
    print(f"      {cls:<22}: {class_weights[i]:.4f}")


print("\n" + "=" * 65)
print("🏗️   STEP 4 — BUILDING EfficientNetB0 MODEL")
print("=" * 65)

def build_model(trainable_base=False):
    """Build EfficientNetB0 model.
       trainable_base=False → Stage 1 (frozen base)
       trainable_base=True  → Stage 2 (fine-tune top layers)
    """
    base = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    base.trainable = trainable_base

    if trainable_base:
        for layer in base.layers[:-30]:
            layer.trainable = False
        print(f"   🔓 Fine-tuning top 30 layers of EfficientNetB0")
    else:
        print(f"   🔒 Base frozen — training custom head only")

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base


print("\n" + "=" * 65)
print("🎯  STEP 5 — STAGE 1: TRAINING CUSTOM HEAD (Base Frozen)")
print("=" * 65)

model, base_model = build_model(trainable_base=False)
model.compile(
    optimizer=Adam(learning_rate=LR_FROZEN),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)
model.summary()

callbacks_stage1 = [
    EarlyStopping(monitor='val_accuracy', patience=8,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=4, min_lr=1e-7, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    CSVLogger(os.path.join(RESULTS_DIR, 'stage1_log.csv'))
]

print(f"\n   🚀 Training Stage 1 — up to {EPOCHS_FROZEN} epochs...")
print(f"   Expected time on M2: ~25-35 minutes\n")

history1 = model.fit(
    train_gen,
    epochs=EPOCHS_FROZEN,
    validation_data=val_gen,
    callbacks=callbacks_stage1,
    class_weight=class_weight_dict,
    verbose=1
)

print("\n   ✅ Stage 1 complete!")

print("\n" + "=" * 65)
print("🔬  STEP 6 — STAGE 2: FINE-TUNING TOP LAYERS")
print("=" * 65)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LR_FINE),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

callbacks_stage2 = [
    EarlyStopping(monitor='val_accuracy', patience=10,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                      patience=5, min_lr=1e-8, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    CSVLogger(os.path.join(RESULTS_DIR, 'stage2_log.csv'))
]

print(f"\n   🔓 Fine-tuning top 30 layers with LR={LR_FINE}")
print(f"   🚀 Training Stage 2 — up to {EPOCHS_FINE} epochs...")
print(f"   Expected time on M2: ~45-60 minutes\n")

history2 = model.fit(
    train_gen,
    epochs=EPOCHS_FINE,
    validation_data=val_gen,
    callbacks=callbacks_stage2,
    class_weight=class_weight_dict,
    verbose=1
)

print("\n   ✅ Stage 2 complete!")

print("\n" + "=" * 65)
print("📈  STEP 7 — SAVING TRAINING HISTORY PLOTS")
print("=" * 65)

def merge_histories(h1, h2, keys):
    """Merge stage1 + stage2 history for clean plotting."""
    merged = {}
    for k in keys:
        merged[k] = h1.history.get(k, []) + h2.history.get(k, [])
    return merged

keys = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
merged = merge_histories(history1, history2, keys)
stage1_len = len(history1.history['accuracy'])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('EfficientNetB0 — Training History', fontsize=16, fontweight='bold', y=1.01)

axes[0].plot(merged['accuracy'],     label='Train Acc',  linewidth=2, color='#2A9D8F')
axes[0].plot(merged['val_accuracy'], label='Val Acc',    linewidth=2, color='#E63946')
axes[0].axvline(x=stage1_len - 1, color='gray', linestyle='--', alpha=0.7, label='Fine-tune starts')
axes[0].set_title('Accuracy', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(merged['loss'],     label='Train Loss', linewidth=2, color='#2A9D8F')
axes[1].plot(merged['val_loss'], label='Val Loss',   linewidth=2, color='#E63946')
axes[1].axvline(x=stage1_len - 1, color='gray', linestyle='--', alpha=0.7, label='Fine-tune starts')
axes[1].set_title('Loss', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   📊 Saved: {RESULTS_DIR}/training_history.png")

print("\n" + "=" * 65)
print("🧪  STEP 8 — FINAL MODEL EVALUATION")
print("=" * 65)

print("   Loading best saved model...")
model = keras.models.load_model(MODEL_SAVE_PATH)

results = {}
for gen, name in [(train_gen, 'Training'), (val_gen, 'Validation'), (test_gen, 'Test')]:
    gen.reset()
    loss, acc, prec, rec = model.evaluate(gen, verbose=0)
    results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'loss': loss}
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    print(f"   {name:<12} → Acc: {acc:.4f} | Prec: {prec:.4f} | "
          f"Rec: {rec:.4f} | F1: {f1:.4f}")

results_df = pd.DataFrame(results).T
results_df.to_csv(os.path.join(RESULTS_DIR, 'model_results.csv'))
print(f"\n   💾 Results saved: {RESULTS_DIR}/model_results.csv")
print(f"   💾 Model saved  : {MODEL_SAVE_PATH}")

print("\n" + "=" * 65)
print("🎉  TRAINING COMPLETE!")
print("=" * 65)
final_acc = results['Test']['accuracy']
print(f"   Final Test Accuracy : {final_acc:.4f} ({final_acc*100:.2f}%)")
print(f"   Model File          : {MODEL_SAVE_PATH}")
print(f"   Results Folder      : {RESULTS_DIR}/")
print()
if final_acc >= 0.88:
    print("   🔥 EXCELLENT — Portfolio-ready accuracy!")
elif final_acc >= 0.80:
    print("   ✅ GOOD — Solid improvement over baseline.")
else:
    print("   ⚠️  Run Stage 2 longer or check dataset path.")

print("\n   ➡️  Next: Run  src/evaluate.py  for full metrics + confusion matrix")
print("=" * 65)

try:
    shutil.rmtree(TMP_BASE)
    print("\n   🧹 Temp files cleaned up.")
except Exception:
    pass