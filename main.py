import cv2
import numpy as np
import torch
from cnn_model import GenderEthnicityModel
from face_detector import extract_faces
from preprocess import preprocess_face

# Set device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():  # Check for Apple Silicon GPU
    device = torch.device("mps")
print(f"Using device: {device}")

# Load pretrained model
model = GenderEthnicityModel()
model.load_state_dict(torch.load("models/gender_ethnicity_model.pth", map_location=device))
model.to(device)
model.eval()  # Set to evaluation mode

def predict_gender_ethnicity(face):
    face = preprocess_face(face)
    face_tensor = torch.from_numpy(face).unsqueeze(0).float().to(device)  # Add batch dimension
    
    with torch.no_grad():
        gender_pred, ethnicity_pred = model(face_tensor)
        
        # Get class with highest probability
        _, gender_idx = torch.max(gender_pred, 1)
        _, ethnicity_idx = torch.max(ethnicity_pred, 1)
        
        gender = "Male" if gender_idx.item() == 0 else "Female"
        ethnicity_labels = ["Asian", "Black", "Hispanic", "Indian", "White"]
        ethnicity = ethnicity_labels[ethnicity_idx.item()]
    
    return gender, ethnicity

# Run on a test image
image_path = "datasets/test/sample.jpg"
faces = extract_faces(image_path)

for i, face in enumerate(faces):
    gender, ethnicity = predict_gender_ethnicity(face)
    print(f"Face {i+1} - Gender: {gender}, Ethnicity: {ethnicity}")
    
    # Optional: display the face with prediction
    face_rgb = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
    cv2.putText(face_rgb, f"{gender}, {ethnicity}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(f"Face {i+1}", face_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()