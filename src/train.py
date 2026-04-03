import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import shutil
import tempfile

print("TensorFlow version:", tf.__version__)

# =============================================================================
# CONFIGURATION — change dataset path here
# =============================================================================

BASE_DATA_DIR = os.getenv("DATASET_PATH", "/Users/smilodon002/Downloads/Alzheimer_Dataset/AugmentedAlzheimerDataset")
MODEL_SAVE_PATH = "model/alzheimer_model.h5"
RESULTS_DIR = "results"

IMG_SIZE = (224, 224)  # EfficientNetB0 needs 224x224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001  # Lower LR for transfer learning

CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("model", exist_ok=True)

print("Configuration ready!")
print(f"Dataset path: {BASE_DATA_DIR}")
print(f"Image size: {IMG_SIZE}")

# =============================================================================
# DATASET EXPLORATION
# =============================================================================

def explore_dataset(data_dir):
    print("\nEXPLORING DATASET...")

    if not os.path.exists(data_dir):
        print(f"Dataset not found: {data_dir}")
        print("Please set DATASET_PATH environment variable or update BASE_DATA_DIR")
        return False, {}

    total_images = 0
    class_counts = {}

    for class_name in CLASS_NAMES:
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg', '.jfif'))]
            class_counts[class_name] = len(images)
            total_images += len(images)
            print(f"  {class_name}: {len(images)} images")
        else:
            print(f"  {class_name}: Directory not found")
            class_counts[class_name] = 0

    print(f"Total images: {total_images}")

    plt.figure(figsize=(10, 6))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    bars = plt.bar(class_counts.keys(), class_counts.values(), color=colors)
    plt.title("Class Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Alzheimer Stage")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    for bar, count in zip(bars, class_counts.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 str(count), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: class_distribution.png")

    return True, class_counts

success, class_counts = explore_dataset(BASE_DATA_DIR)
if not success:
    exit()

# =============================================================================
# DATA SPLITTING — 70 / 15 / 15
# =============================================================================

def create_data_split(data_dir):
    print("\nCreating 70-15-15 split...")
    temp_base = tempfile.mkdtemp()
    train_dir = os.path.join(temp_base, 'train')
    val_dir   = os.path.join(temp_base, 'val')
    test_dir  = os.path.join(temp_base, 'test')

    for split_dir in [train_dir, val_dir, test_dir]:
        for class_name in CLASS_NAMES:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

    for class_name in CLASS_NAMES:
        class_path = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg', '.jfif'))]

        if len(images) == 0:
            continue

        train_imgs, temp_imgs = train_test_split(images, test_size=0.30, random_state=42)
        val_imgs, test_imgs   = train_test_split(temp_imgs, test_size=0.50, random_state=42)

        for img in train_imgs:
            shutil.copy2(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in val_imgs:
            shutil.copy2(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))
        for img in test_imgs:
            shutil.copy2(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

        print(f"  {class_name}: {len(train_imgs)} train | {len(val_imgs)} val | {len(test_imgs)} test")

    print("Split created!")
    return train_dir, val_dir, test_dir

TRAIN_DIR, VAL_DIR, TEST_DIR = create_data_split(BASE_DATA_DIR)

# =============================================================================
# DATA GENERATORS
# =============================================================================

print("\nSetting up data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)

val_datagen  = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True, seed=42
)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

print(f"Training samples:   {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples:       {test_generator.samples}")

# =============================================================================
# CLASS WEIGHTS
# =============================================================================

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print("\nClass weights:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {class_weights[i]:.4f}")

# =============================================================================
# MODEL — EfficientNetB0 Transfer Learning
# =============================================================================

print("\nBuilding EfficientNetB0 model...")

def build_efficientnet_model():
    # Load EfficientNetB0 pre-trained on ImageNet
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Phase 1 — Freeze base model, train only classifier
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

model, base_model = build_efficientnet_model()
print("EfficientNetB0 model built!")
print(f"Total parameters: {model.count_params():,}")

# =============================================================================
# CALLBACKS
# =============================================================================

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=1e-8,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# =============================================================================
# PHASE 1 TRAINING — Train classifier only (frozen base)
# =============================================================================

print("\n" + "="*60)
print("PHASE 1: Training classifier head (base frozen)")
print("Expected time: 20-30 minutes")
print("="*60)

history_phase1 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

print("Phase 1 complete!")
phase1_best_val = max(history_phase1.history['val_accuracy'])
print(f"Best validation accuracy (Phase 1): {phase1_best_val:.4f}")

# =============================================================================
# PHASE 2 TRAINING — Fine-tune top layers of base model
# =============================================================================

print("\n" + "="*60)
print("PHASE 2: Fine-tuning top layers of EfficientNetB0")
print("Expected time: 20-40 minutes")
print("="*60)

# Unfreeze top 30 layers of base model
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with very low learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=1e-9,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history_phase2 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=callbacks_phase2,
    class_weight=class_weight_dict,
    verbose=1
)

print("Phase 2 complete!")
phase2_best_val = max(history_phase2.history['val_accuracy'])
print(f"Best validation accuracy (Phase 2): {phase2_best_val:.4f}")

# =============================================================================
# COMBINE HISTORY
# =============================================================================

combined_history = {
    'accuracy':     history_phase1.history['accuracy']     + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    'loss':         history_phase1.history['loss']         + history_phase2.history['loss'],
    'val_loss':     history_phase1.history['val_loss']     + history_phase2.history['val_loss'],
}

# =============================================================================
# EVALUATION
# =============================================================================

print("\nEvaluating model on all splits...")
model = tf.keras.models.load_model(MODEL_SAVE_PATH)

results = {}
for generator, name in [(train_generator, "Training"),
                        (val_generator, "Validation"),
                        (test_generator, "Test")]:
    generator.reset()
    loss, accuracy = model.evaluate(generator, verbose=0)
    results[name] = {'accuracy': accuracy, 'loss': loss}
    print(f"  {name:12} → Accuracy: {accuracy:.4f} | Loss: {loss:.4f}")

# =============================================================================
# DETAILED METRICS
# =============================================================================

test_generator.reset()
predictions     = model.predict(test_generator, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
true_classes      = test_generator.classes

f1 = f1_score(true_classes, predicted_classes, average='weighted')
print(f"\nF1-Score:  {f1:.4f} ({f1*100:.2f}%)")

try:
    y_true_onehot = tf.keras.utils.to_categorical(true_classes, num_classes=4)
    auc_roc = roc_auc_score(y_true_onehot, predictions, multi_class='ovr')
    print(f"AUC-ROC:   {auc_roc:.4f} ({auc_roc*100:.2f}%)")
except Exception as e:
    auc_roc = 0.0
    print(f"AUC-ROC calculation skipped: {e}")

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes,
                             target_names=CLASS_NAMES, digits=4))

# =============================================================================
# VISUALIZATIONS
# =============================================================================

# Training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(combined_history['accuracy'],     label='Train Accuracy',  linewidth=2)
plt.plot(combined_history['val_accuracy'], label='Val Accuracy',    linewidth=2)
plt.axvline(x=len(history_phase1.history['accuracy']),
            color='gray', linestyle='--', alpha=0.7, label='Fine-tune starts')
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(combined_history['loss'],     label='Train Loss', linewidth=2)
plt.plot(combined_history['val_loss'], label='Val Loss',   linewidth=2)
plt.axvline(x=len(history_phase1.history['loss']),
            color='gray', linestyle='--', alpha=0.7, label='Fine-tune starts')
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: training_history.png")

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix — Test Set', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix.png")

# =============================================================================
# SAVE RESULTS
# =============================================================================

results_summary = {
    'Model':    'EfficientNetB0 (Transfer Learning)',
    'Accuracy': f"{results['Test']['accuracy']*100:.2f}%",
    'F1-Score': f"{f1*100:.2f}%",
    'AUC-ROC':  f"{auc_roc*100:.2f}%",
}
pd.DataFrame([results_summary]).to_csv(
    os.path.join(RESULTS_DIR, 'model_results.csv'), index=False)
print("Saved: model_results.csv")

# =============================================================================
# CLEANUP
# =============================================================================

try:
    shutil.rmtree(os.path.dirname(TRAIN_DIR))
    print("\nTemporary files cleaned up!")
except:
    pass

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Final Test Accuracy : {results['Test']['accuracy']*100:.2f}%")
print(f"F1-Score            : {f1*100:.2f}%")
print(f"AUC-ROC             : {auc_roc*100:.2f}%")
print(f"Model saved         : {MODEL_SAVE_PATH}")
print("="*60)

prev_accuracy = 77.13
new_accuracy  = results['Test']['accuracy'] * 100
improvement   = new_accuracy - prev_accuracy
print(f"\nPrevious accuracy (Custom CNN): {prev_accuracy}%")
print(f"New accuracy (EfficientNetB0):  {new_accuracy:.2f}%")
print(f"Improvement:                    +{improvement:.2f}%")
print("="*60)