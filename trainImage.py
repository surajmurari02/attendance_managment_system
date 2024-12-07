import os
import cv2
import numpy as np
import re
from PIL import Image

# Function to get images and labels for training
def getImagesAndLables(trainimage_path):
    imagePaths = [
        os.path.join(trainimage_path, f) for f in os.listdir(trainimage_path) if os.path.isfile(os.path.join(trainimage_path, f))
    ]
    
    faces = []
    Ids = []
    
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert("L")  # Convert image to grayscale
            imageNp = np.array(pilImage, "uint8")
            
            # Use regular expression to extract the numerical ID from filename (assuming it's the first number)
            match = re.search(r"(\d+)", os.path.split(imagePath)[-1])
            if match:
                Id = int(match.group(1))  # Extract and convert the ID to integer
                faces.append(imageNp)
                Ids.append(Id)
            else:
                print(f"Could not extract ID from {imagePath}")
        except Exception as e:
            print(f"Error processing {imagePath}: {e}")
    
    return faces, Ids

# Train the face recognizer model
def TrainImage(haarcasecade_path, trainimage_path, trainimagelabel_path, message, text_to_speech):
    # Initialize recognizer and face detector
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(haarcasecade_path)
    
    # Get images and labels
    faces, Id = getImagesAndLables(trainimage_path)
    
    if len(faces) == 0 or len(Id) == 0:
        message.configure(text="No faces found for training.")
        text_to_speech("No faces found for training.")
        return

    # Train the model
    recognizer.train(faces, np.array(Id))

    # Save the trained model
    recognizer.save(trainimagelabel_path)
    
    res = "Image Trained successfully"
    message.configure(text=res)
    text_to_speech(res)
