import face_recognition
import os
import cv2
import numpy as np
from typing import List, Tuple

class FaceRecognitionSystem:
    def __init__(self, known_faces_dir: str, unknown_faces_dir: str, 
                 tolerance: float = 0.6, model: str = "hog"):  
        self.known_faces_dir = known_faces_dir
        self.unknown_faces_dir = unknown_faces_dir
        self.tolerance = tolerance
        self.model = model
        self.frame_thickness = 3
        self.font_thickness = 2
        
        self.known_faces = []
        self.known_names = []
        
    def process_unknown_faces(self) -> None:
        print("Processing unknown faces...")
        
        for filename in os.listdir(self.unknown_faces_dir):
            try:
                print(f"\nProcessing {filename}...")
                image_path = os.path.join(self.unknown_faces_dir, filename)
                
                # Load image
                print("Loading image...")
                image = face_recognition.load_image_file(image_path)
                print(f"Image loaded successfully. Shape: {image.shape}")
                
                # Find faces
                print("Finding faces...")
                locations = face_recognition.face_locations(image, model=self.model)
                print(f"Found {len(locations)} faces")
                
                encodings = face_recognition.face_encodings(image, locations)
                
                # Convert image
                print("Converting image for display...")
                display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Process faces
                for face_encoding, face_location in zip(encodings, locations):
                    matches = face_recognition.compare_faces(self.known_faces, face_encoding, self.tolerance)
                    
                    if True in matches:
                        match_index = matches.index(True)
                        name = self.known_names[match_index]
                        print(f"Found match: {name}")
                        
                        # Draw rectangle
                        top, right, bottom, left = face_location
                        cv2.rectangle(display_image, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.rectangle(display_image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                        cv2.putText(display_image, name, (left + 6, bottom - 6), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Save the output image instead of displaying
                output_path = f"output_{filename}"
                print(f"Saving result to {output_path}")
                cv2.imwrite(output_path, display_image)
                print(f"Result saved to {output_path}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    def load_known_faces(self):
        print("Loading known faces...")
        for name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, name)
            if not os.path.isdir(person_dir):
                continue
                
            print(f"Loading faces for {name}...")
            for filename in os.listdir(person_dir):
                try:
                    image_path = os.path.join(person_dir, filename)
                    print(f"  Processing {filename}")
                    
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if len(encodings) > 0:
                        self.known_faces.append(encodings[0])
                        self.known_names.append(name)
                        print(f"  Successfully loaded face from {filename}")
                    else:
                        print(f"  Warning: No face found in {filename}")
                        
                except Exception as e:
                    print(f"  Error processing {filename}: {str(e)}")
        
        print(f"Loaded {len(self.known_faces)} known faces")

def main():
    try:
        # Initialize face recognition system
        system = FaceRecognitionSystem(
            known_faces_dir="known_faces",
            unknown_faces_dir="unknown_faces",
            tolerance=0.6,  
            model="hog"    
        )
        # Load and process faces
        system.load_known_faces()
        system.process_unknown_faces()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()