import os
import medmnist
from medmnist import INFO
import numpy as np
from PIL import Image

# Define target paths
base_dir = "data"
sets = ['train', 'val', 'test']
classes = ['0_no_dr', '1_dr']

# Create directories if they don't exist
for s in sets:
    for c in classes:
        os.makedirs(os.path.join(base_dir, s, c), exist_ok=True)

# RetinaMNIST dataset details
data_flag = 'retinamnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

print(f"Downloading and extracting {data_flag}...")

# Download subsets
train_dataset = DataClass(split='train', download=True)
val_dataset = DataClass(split='val', download=True)
test_dataset = DataClass(split='test', download=True)

def process_and_save(dataset, split_name):
    print(f"Processing {split_name} set...")
    for idx, (img, label) in enumerate(dataset):
        # MedMNIST returns (PIL.Image, numpy.ndarray)
        # For RetinaMNIST, labels are: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative
        # Binary target: 0 -> 0 (No DR), [1,2,3,4] -> 1 (DR)
        original_label = label[0]
        binary_label = 0 if original_label == 0 else 1
        
        # Determine class directory
        class_dir = classes[binary_label]
        
        # Resize image for standard CNNs (MedMNIST images are 28x28, we upscale it slightly to 128x128 to let standard CNN patterns work nicely and generate decent Grad-CAMs)
        img_resized = img.resize((128, 128), Image.BILINEAR)
        
        # Create filename
        filename = f"{split_name}_{idx}.jpg"
        filepath = os.path.join(base_dir, split_name, class_dir, filename)
        
        # Save image
        img_resized.save(filepath)

# Process all subsets
process_and_save(train_dataset, "train")
process_and_save(val_dataset, "val")
# We'll put test images into val to maximize our validation set (this is just for demonstration purposes)
process_and_save(test_dataset, "val")

print("Data preparation complete.")
print(f"Train/0_no_dr: {len(os.listdir(os.path.join(base_dir, 'train', '0_no_dr')))} images")
print(f"Train/1_dr: {len(os.listdir(os.path.join(base_dir, 'train', '1_dr')))} images")
print(f"Val/0_no_dr: {len(os.listdir(os.path.join(base_dir, 'val', '0_no_dr')))} images")
print(f"Val/1_dr: {len(os.listdir(os.path.join(base_dir, 'val', '1_dr')))} images")
