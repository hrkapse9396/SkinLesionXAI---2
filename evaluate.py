import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.data_loader import get_data_generators
from src.model import build_model
from src import config
import tensorflow as tf

def evaluate_model():
    # 1. Get Data (We only need validation data here)
    _, val_gen = get_data_generators()
    
    # IMPORTANT: Reset generator to start and turn off shuffling for evaluation
    val_gen.reset()
    val_gen.shuffle = False
    
    # 2. Load Model
    print("Loading Model...")
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    
    # 3. Predict on all validation images
    print("Generating predictions... (This may take a minute)")
    Y_pred = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    # Get true labels
    y_true = val_gen.classes
    
    # 4. Classification Report
    class_labels = list(val_gen.class_indices.keys())
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_labels))
    
    # 5. Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    evaluate_model()