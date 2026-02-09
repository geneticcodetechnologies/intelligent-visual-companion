import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once (IMPORTANT for performance)
MODEL_PATH = "model/saved_model"
detect_fn = tf.saved_model.load(MODEL_PATH)

def load_image_into_numpy_array(image):
    return np.array(image)

def run_inference(image: Image.Image):
    image_np = load_image_into_numpy_array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    return {
        "boxes": detections["detection_boxes"][0].numpy().tolist(),
        "scores": detections["detection_scores"][0].numpy().tolist(),
        "classes": detections["detection_classes"][0].numpy().astype(int).tolist()
    }
