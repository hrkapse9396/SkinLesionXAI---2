import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import os
import random
from tensorflow.keras.preprocessing import image
from src import config, xai_utils

def test_random_image():
    # 1. Load the trained model
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("Error: Model file not found. Run train.py first!")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

    # 2. Pick a random image from data folder
    all_paths = glob.glob(os.path.join(config.DATA_DIR, '*', '*.jpg'))
    random_path = random.choice(all_paths)
    print(f"Analyzing Image: {random_path}")

    # 3. Preprocess the image
    img = image.load_img(random_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Make batch of 1
    img_array /= 255.0

    # 4. Predict
    preds = model.predict(img_array)
    print(f"Prediction: {preds}")

    # 5. Generate Heatmap
    # For ResNet50, the last conv layer is usually 'conv5_block3_out'
    heatmap = xai_utils.make_gradcam_heatmap(img_array, model, 'conv5_block3_out')

    # 6. Display Result
    xai_utils.display_gradcam(random_path, heatmap)

if __name__ == "__main__":
    test_random_image()