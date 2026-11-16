from pathlib import Path
import numpy as np
from PIL import Image

BASE_DIR=Path(__file__).resolve().parent
MODEL_PATH=BASE_DIR/"models"/"pneumonia_classifier.keras"
IMAGE_SIZE=(224,224)

def predict_image(image_path):
    if not MODEL_PATH.exists():
        raise RuntimeError("Trained model not found. Download the dataset and run python train.py as described in README.md.")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is not installed. Install requirements.txt first.") from exc
    model=tf.keras.models.load_model(MODEL_PATH)
    image=Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    array=np.asarray(image,dtype=np.float32)/255.0
    # The model has one sigmoid output: probability of the second class, PNEUMONIA.
    pneumonia_probability=float(model.predict(np.expand_dims(array,axis=0),verbose=0).reshape(-1)[0])
    prediction="PNEUMONIA" if pneumonia_probability>=0.5 else "NORMAL"
    confidence=pneumonia_probability if prediction=="PNEUMONIA" else 1.0-pneumonia_probability
    return {"prediction":prediction,"confidence":round(confidence*100,2),"model_version":"MobileNetV2-1.0"}
