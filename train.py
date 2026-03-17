from src.data_loader import get_data_generators
from src.model import build_model
from src import config
from tensorflow.keras.callbacks import ModelCheckpoint

if __name__ == "__main__":
    # 1. Prepare Data
    train_gen, val_gen = get_data_generators()

    # 2. Build Model
    model = build_model()
    
    # 3. Save the best model during training
    checkpoint = ModelCheckpoint(config.MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

    # 4. Train
    print("Starting training... (This may take time)")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=[checkpoint]
    )
    print("Training finished!")