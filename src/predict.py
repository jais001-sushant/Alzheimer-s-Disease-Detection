import tensorflow as tf
import numpy as np
from PIL import Image
import os

CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
IMG_SIZE    = (224, 224)

CLASS_INFO = {
    'NonDemented': {
        'label':       'No Alzheimer\'s Detected',
        'color':       '#2ecc71',
        'emoji':       '🟢',
        'description': 'The MRI scan shows no significant signs of cognitive impairment. Brain structure appears normal.',
        'advice':      'Continue regular health checkups. Maintain a healthy lifestyle with exercise and mental stimulation.',
        'severity':    0
    },
    'VeryMildDemented': {
        'label':       'Very Mild Cognitive Decline',
        'color':       '#f1c40f',
        'emoji':       '🟡',
        'description': 'Very subtle signs of cognitive decline detected. Early stage — minimal impact on daily life.',
        'advice':      'Consult a neurologist for comprehensive evaluation. Early intervention can significantly slow progression.',
        'severity':    1
    },
    'MildDemented': {
        'label':       'Mild Cognitive Impairment',
        'color':       '#e67e22',
        'emoji':       '🟠',
        'description': 'Moderate signs of Alzheimer\'s detected. May experience memory difficulties and cognitive challenges.',
        'advice':      'Seek specialist medical attention. Treatment and lifestyle changes at this stage can improve quality of life.',
        'severity':    2
    },
    'ModerateDemented': {
        'label':       'Moderate Alzheimer\'s Disease',
        'color':       '#e74c3c',
        'emoji':       '🔴',
        'description': 'Significant signs of Alzheimer\'s disease detected. Requires immediate medical attention and care planning.',
        'advice':      'Immediate consultation with a neurologist is strongly recommended. Care planning and support are essential.',
        'severity':    3
    }
}

def load_model(model_path="model/alzheimer_model.h5"):
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(pil_img):
    img = pil_img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, pil_img):
    img_array   = preprocess_image(pil_img)
    predictions = model.predict(img_array, verbose=0)[0]

    predicted_idx   = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = float(predictions[predicted_idx])

    all_confidences = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        'predicted_class': predicted_class,
        'confidence':      confidence,
        'all_confidences': all_confidences,
        'class_info':      CLASS_INFO[predicted_class]
    }