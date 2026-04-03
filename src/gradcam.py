# =============================================================================
# ALZHEIMER'S DETECTION — Phase 2: Grad-CAM Explainability
# Author: Sushant Jaiswal | UPES Dehradun
# Generates visual heatmaps showing WHERE the model looks in the MRI
# =============================================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from PIL import Image

print("=" * 65)
print("🔥  ALZHEIMER'S DETECTION — Grad-CAM Explainability")
print("=" * 65)

# =============================================================================
# CONFIGURATION
# =============================================================================

IMG_SIZE    = (224, 224)
CLASS_NAMES = ['MildDemented', 'ModerateDemented',
               'NonDemented', 'VeryMildDemented']

MODEL_PATH  = "best_alzheimer_efficientnet.keras"
RESULTS_DIR = "results"
GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam")
os.makedirs(GRADCAM_DIR, exist_ok=True)

# Staging descriptions for UI display
STAGE_INFO = {
    'NonDemented':      {'level': 0, 'color': '#2A9D8F',
                         'desc': 'No signs of cognitive impairment detected.'},
    'VeryMildDemented': {'level': 1, 'color': '#E9C46A',
                         'desc': 'Very early stage. Subtle memory lapses possible.'},
    'MildDemented':     {'level': 2, 'color': '#F4A261',
                         'desc': 'Mild cognitive impairment. Consult a specialist.'},
    'ModerateDemented': {'level': 3, 'color': '#E63946',
                         'desc': 'Moderate dementia detected. Immediate attention needed.'}
}

# =============================================================================
# LOAD MODEL
# =============================================================================

print(f"\n   Loading model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print("❌  Model not found! Run train.py first.")
    exit(1)

model = keras.models.load_model(MODEL_PATH)
print("   ✅ Model loaded!")

# =============================================================================
# GRAD-CAM CORE FUNCTIONS
# =============================================================================

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for a given image.
    
    Args:
        model           : Trained Keras model
        img_array       : Preprocessed image (1, H, W, 3)
        last_conv_layer_name : Name of last conv layer in base model
        pred_index      : Class index to explain (None = predicted class)
    
    Returns:
        heatmap (H, W) numpy array with values in [0, 1]
    """
    # Build gradient model: input → last conv layer + predictions
    grad_model = keras.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(last_conv_layer_name).output,
                   model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradients of predicted class w.r.t. last conv output
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight conv outputs by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_gradcam(original_img_path, heatmap, alpha=0.45):
    """
    Overlay Grad-CAM heatmap on original MRI image.
    
    Returns:
        superimposed image as numpy array (H, W, 3) uint8
    """
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8   = np.uint8(255 * heatmap_resized)

    # Apply colormap
    jet = cm.get_cmap("jet")
    jet_colors  = jet(np.arange(256))[:, :3]
    heatmap_rgb = jet_colors[heatmap_uint8]
    heatmap_rgb = np.uint8(heatmap_rgb * 255)

    # Superimpose
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_rgb, alpha, 0)
    return superimposed, img


def preprocess_image(image_path):
    """Load and preprocess a single image for model input."""
    img = keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


def find_last_conv_layer(model):
    """Auto-detect the last convolutional layer name in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model):
            # It's the EfficientNetB0 sub-model
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, keras.layers.Conv2D):
                    return layer.name, sub_layer.name
    return None, None


def generate_gradcam(image_path, model, save_path=None, show=False):
    """
    Full Grad-CAM pipeline for one image.
    
    Returns:
        predicted_class, confidence, all_probs, superimposed_img
    """
    if not os.path.exists(image_path):
        print(f"   ❌ Image not found: {image_path}")
        return None

    # Preprocess
    img_array = preprocess_image(image_path)

    # Predict
    preds       = model.predict(img_array, verbose=0)[0]
    pred_idx    = np.argmax(preds)
    pred_class  = CLASS_NAMES[pred_idx]
    confidence  = preds[pred_idx]

    # Find last conv layer in EfficientNetB0
    # EfficientNetB0 last conv layer is 'top_conv'
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model):
            # sub-model (EfficientNetB0)
            for sub in reversed(layer.layers):
                if 'conv' in sub.name.lower() and isinstance(sub, keras.layers.Conv2D):
                    last_conv = sub.name
                    break
            if last_conv:
                break

    if last_conv is None:
        last_conv = 'top_conv'  # EfficientNetB0 default

    # Build grad model targeting the sub-model's conv layer
    # We need to build it differently for nested models
    base_submodel = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base_submodel = layer
            break

    if base_submodel is not None:
        try:
            grad_model = keras.Model(
                inputs  = model.inputs,
                outputs = [base_submodel.get_layer('top_conv').output,
                           model.output]
            )

            with tf.GradientTape() as tape:
                conv_out, predictions = grad_model(img_array)
                class_channel = predictions[:, pred_idx]

            grads = tape.gradient(class_channel, conv_out)
            pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = conv_out[0] @ pooled[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
            heatmap = heatmap.numpy()
        except Exception as e:
            print(f"   ⚠️  Grad-CAM fallback: {e}")
            heatmap = np.zeros(IMG_SIZE)
    else:
        heatmap = np.zeros(IMG_SIZE)

    # Overlay
    superimposed, original = overlay_gradcam(image_path, heatmap)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    stage = STAGE_INFO.get(pred_class, {})

    fig.suptitle(
        f'Grad-CAM Analysis  |  Predicted: {pred_class}  |  '
        f'Confidence: {confidence*100:.1f}%',
        fontsize=14, fontweight='bold', color=stage.get('color', 'black')
    )

    axes[0].imshow(original)
    axes[0].set_title('Original MRI', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap\n(Red = High attention)',
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(superimposed)
    axes[2].set_title('Overlay on MRI', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Confidence bars
    ax_bar = fig.add_axes([0.05, -0.12, 0.9, 0.1])
    bar_colors = [STAGE_INFO[c]['color'] for c in CLASS_NAMES]
    bars = ax_bar.barh(CLASS_NAMES, preds * 100, color=bar_colors,
                       edgecolor='white', height=0.6)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel('Confidence (%)', fontsize=10)
    ax_bar.set_title('Class Probabilities', fontsize=11, fontweight='bold')
    for bar, prob in zip(bars, preds):
        ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{prob*100:.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")

    if show:
        plt.show()

    plt.close()

    return pred_class, confidence, preds, superimposed


# =============================================================================
# GENERATE GRADCAM FOR SAMPLE IMAGES
# =============================================================================

print("\n" + "=" * 65)
print("🔥  GENERATING GRAD-CAM HEATMAPS FOR SAMPLE IMAGES")
print("=" * 65)

BASE_DATA_DIR = os.getenv(
    "DATASET_PATH",
    "/Users/smilodon002/Downloads/Alzheimer_Dataset/AugmentedAlzheimerDataset"
)

print("\n   Generating Grad-CAM for 1 sample per class...\n")

for cls_name in CLASS_NAMES:
    cls_path = os.path.join(BASE_DATA_DIR, cls_name)
    if not os.path.exists(cls_path):
        print(f"   ⚠️  Skipping {cls_name} — path not found")
        continue

    imgs = [f for f in os.listdir(cls_path)
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.jfif'))]
    if not imgs:
        continue

    sample_path = os.path.join(cls_path, imgs[0])
    save_path   = os.path.join(GRADCAM_DIR, f'gradcam_{cls_name}.png')

    print(f"   🧠 Processing: {cls_name}")
    result = generate_gradcam(sample_path, model, save_path=save_path)
    if result:
        pred_class, confidence, probs, _ = result
        print(f"      Predicted : {pred_class}  ({confidence*100:.1f}%)")
        print(f"      True Class: {cls_name}")
        print()

print("=" * 65)
print("✅  Grad-CAM generation complete!")
print(f"   Heatmaps saved in: {GRADCAM_DIR}/")
print("   ➡️  Next: Run  app/app.py  to launch Streamlit web app")
print("=" * 65)