import face_recognition
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class FaceRecognition:
    def __init__(self, model='hog', tolerance=0.6):
        """
        Initialize Face Recognition module
        
        Args:
            model (str): 'hog' for CPU, 'cnn' for GPU (faster but requires CUDA)
            tolerance (float): How much distance between faces to consider a match (0-1)
        """
        self.model = model
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
    
    def load_person_faces(self, person_name: str, directory_path: str):
        """
        Load all face images of a person from a directory
        
        Args:
            person_name (str): Name of the person
            directory_path (str): Path to directory containing face images
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        path = Path(directory_path)
        
        count = 0
        for image_file in path.iterdir():
            if image_file.suffix.lower() in image_extensions:
                try:
                    image = face_recognition.load_image_file(str(image_file))
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(person_name)
                        count += 1
                        logger.info(f"Loaded face {count} for {person_name}")
                except Exception as e:
                    logger.error(f"Error loading {image_file}: {e}")
        
        if count == 0:
            logger.warning(f"No faces found for {person_name} in {directory_path}")
    
    def identify_faces(self, image: np.ndarray, tolerance: float = None) -> List[Dict]:
        """
        Identify faces in an image
        
        Args:
            image (np.ndarray): RGB image array
            tolerance (float): Optional override for tolerance
        
        Returns:
            List of dictionaries with identification results
        """
        if tolerance is None:
            tolerance = self.tolerance
        
        # Find all faces and encodings in the image
        face_locations = face_recognition.face_locations(image, model=self.model)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        identifications = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            
            # Find the best match
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                best_distance = distances[best_match_index]
                
                if best_distance <= tolerance:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - best_distance  # Convert distance to confidence
                else:
                    name = "Unknown"
                    confidence = 0.0
            else:
                name = "Unknown"
                confidence = 0.0
            
            identifications.append({
                'face_location': (top, right, bottom, left),
                'name': name,
                'confidence': confidence,
                'distance': best_distance if len(distances) > 0 else 1.0
            })
        
        return identifications
    
    def add_face_encoding(self, person_name: str, image: np.ndarray):
        """
        Manually add a face encoding
        
        Args:
            person_name (str): Name of the person
            image (np.ndarray): RGB image array containing the face
        """
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(person_name)
            logger.info(f"Added face encoding for {person_name}")
        else:
            logger.warning(f"No face found in image for {person_name}")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about loaded faces
        
        Returns:
            Dictionary with statistics
        """
        unique_people = set(self.known_face_names)
        return {
            'total_encodings': len(self.known_face_encodings),
            'unique_people': len(unique_people),
            'people': list(unique_people),
            'model': self.model,
            'tolerance': self.tolerance
        }
