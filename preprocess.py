import os
import cv2
import numpy as np
from tqdm import tqdm

DATASET_PATH = "datasets/UTKFace/"
IMAGE_SIZE = (128, 128)  

image_data = []
gender_labels = []
ethnicity_labels = []
# Read all image files
for filename in tqdm(os.listdir(DATASET_PATH)):
    if filename.endswith(".jpg"):  # Process only image files
        try:
            # Extract labels from filename
            parts = filename.split("_")
            age = int(parts[0])  # Age (not used)
            gender = int(parts[1])  # Gender (0 = male, 1 = female)
            ethnicity = int(parts[2])  # Ethnicity (0-4)
            #image is loading here
            img_path = os.path.join(DATASET_PATH, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMAGE_SIZE)  # Resize to (128, 128)
            img = img.astype("float32") / 255.0  # Normalize to [0,1]

            # Append to lists
            image_data.append(img)
            gender_labels.append(gender)
            ethnicity_labels.append(ethnicity)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# convert to numpy arrays
image_data = np.array(image_data)
gender_labels = np.array(gender_labels)
ethnicity_labels = np.array(ethnicity_labels)

#.npy files
np.save("datasets/train_images.npy", image_data)
np.save("datasets/gender_labels.npy", gender_labels)
np.save("datasets/ethnicity_labels.npy", ethnicity_labels)

print("Dataset preprocessing complete! Files saved:")
print("   - datasets/train_images.npy")
print("   - datasets/gender_labels.npy")
print("   - datasets/ethnicity_labels.npy")

def preprocess_face(face):
    """Preprocess a single face for model prediction"""
    face = cv2.resize(face, IMAGE_SIZE)
    face = face.astype("float32") / 255.0
    # Convert to channel-first format for PyTorch (H,W,C) -> (C,H,W)
    face = np.transpose(face, (2, 0, 1))
    return face