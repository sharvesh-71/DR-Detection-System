import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import math

base_dir = "data"
model_path = "model/dr_model.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def get_class_weights(train_dir):
    class_0_dir = os.path.join(train_dir, '0_no_dr')
    class_1_dir = os.path.join(train_dir, '1_dr')
    
    if not os.path.exists(class_0_dir) or not os.path.exists(class_1_dir):
        return {0: 1.0, 1: 1.0}
        
    num_0 = len(os.listdir(class_0_dir))
    num_1 = len(os.listdir(class_1_dir))
    total = num_0 + num_1
    
    if total == 0:
        return {0: 1.0, 1: 1.0}
        
    weight_0 = (1 / num_0) * (total / 2.0)
    weight_1 = (1 / num_1) * (total / 2.0)
    
    print(f"Computed Class Weights: 0: {weight_0:.2f}, 1: {weight_1:.2f}")
    return {0: weight_0, 1: weight_1}

def create_model(apply_augmentation=False):
    model = models.Sequential()
    
    if apply_augmentation:
        model.add(layers.RandomFlip("horizontal_and_vertical", input_shape=(128, 128, 3)))
        model.add(layers.RandomRotation(0.2))
        model.add(layers.RandomZoom(0.2))
        # Rescale happens after input layer for augmentation
        model.add(layers.Rescaling(1./255))
    else:
        model.add(layers.Input(shape=(128, 128, 3)))
        model.add(layers.Rescaling(1./255))
    
    # We name the last conv layer for Grad-CAM
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', name='last_conv_layer'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate(apply_augmentation, epochs=5):
    print(f"--- Training Model (Augmentation: {apply_augmentation}, Epochs: {epochs}) ---")
    
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )
    
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        shuffle=False,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )
    
    class_weights = get_class_weights(train_dir)
    model = create_model(apply_augmentation)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weights,
        verbose=1
    )
    
    val_loss, val_acc = model.evaluate(val_dataset)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    return model, val_acc

def main():
    target_accuracy = 0.65
    max_attempts = 3
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Attempt {attempt}/{max_attempts} ---")
        
        apply_aug = (attempt >= 2)
        epochs = 10 if attempt == 1 else 15  # 10 epochs initially, more later
        
        model, val_acc = train_and_evaluate(apply_augmentation=apply_aug, epochs=epochs)
        
        if val_acc >= target_accuracy:
            print(f"Target accuracy achieved! ({val_acc:.4f} >= {target_accuracy})")
            model.save(model_path)
            print(f"Model saved to {model_path}")
            break
        else:
            print(f"Accuracy ({val_acc:.4f}) below target ({target_accuracy}). Retrying...")
            
        if attempt == max_attempts:
            print("Max attempts reached. Saving final model.")
            model.save(model_path)

if __name__ == "__main__":
    main()
