import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import json
from PIL import Image
import warnings
import logging
import tf_keras

warnings.filterwarnings('ignore')
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255.0
    return image.numpy()
    
def predict_class(image_path, model, top_k):
    image = np.asarray(Image.open(image_path))
    processed_image = process_image(image)
    
    img_batch = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(img_batch).flatten()

    top_indices = predictions.argsort()[-top_k:][::-1]
    top_probs = predictions[top_indices]
    return top_probs, top_indices

def parse_arguments():
    parser = argparse.ArgumentParser(description="Flower Classification")
    parser.add_argument('image_path', help="Path to the image")
    parser.add_argument('saved_model', help="Path to the saved model")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top predictions to display (default: 5)")
    parser.add_argument('--category_names', help="Path to category names JSON file (mappinng)")
    return parser.parse_args()

def display_results(probs, classes, class_names=None):
    print("\n\n\n------------- Prediction Results ------------")
    for i, (prob, cls) in enumerate(zip(probs, classes)):
        class_label = class_names.get(str(cls), f"Class {cls}") if class_names else f"Class {cls}"
        print(f"Rank {i+1}:")
        print(f"  Class: {class_label}")
        print(f"  Probability: {prob:.2%}")
        print("-" * 30)

def load_class_names(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading category names: {e}")
        return None

def main():
    args = parse_arguments()
    try:
        model = tf_keras.models.load_model(args.saved_model, custom_objects={'KerasLayer': hub.KerasLayer})
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    probs, classes = predict_class(args.image_path, model, args.top_k)
    class_names = None
    if args.category_names:
        class_names = load_class_names(args.category_names)

    display_results(probs, classes, class_names)

if __name__ == "__main__":
    main()
