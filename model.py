from pathlib import Path
import numpy as np
from PIL import Image
BASE_DIR=Path(__file__).resolve().parent
MODEL_PATH=BASE_DIR/"models"/"pneumonia_classifier.keras"
IMAGE_SIZE=(224,224)
def predict_image(image_path):
    if not MODEL_PATH.exists():
        raise RuntimeError("Trained model not found. Download the dataset and run python train.py as described in README.md.")
    try: import tensorflow as tf
    except ImportError as exc: raise RuntimeError("TensorFlow is not installed. Install requirements.txt first.") from exc
    model=tf.keras.models.load_model(MODEL_PATH)
    image=Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    array=np.asarray(image,dtype=np.float32)/255.0
    probabilities=model.predict(np.expand_dims(array,axis=0),verbose=0)[0]
    index=int(np.argmax(probabilities)); labels=["NORMAL","PNEUMONIA"]
    return {"prediction":labels[index],"confidence":round(float(probabilities[index])*100,2),"model_version":"MobileNetV2-1.0"}
