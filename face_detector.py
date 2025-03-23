import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch

def extract_faces(image_path):
    # Create MTCNN detector from facenet_pytorch
    device = torch.device('cpu')
    detector = MTCNN(keep_all=True, device=device)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces using the correct method for facenet_pytorch MTCNN
    boxes, _ = detector.detect(image_rgb)
    face_crops = []

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            cropped_face = image_rgb[y1:y2, x1:x2]
            cropped_face = cv2.resize(cropped_face, (128, 128))  # resize cnn input
            face_crops.append(cropped_face)

    return face_crops