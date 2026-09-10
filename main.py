import cv2
import numpy as np
from face_recognition_module import FaceRecognition
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceIdentificationApp:
    def __init__(self, known_faces_dir='known_faces'):
        """
        Initialize the Face Identification Application
        
        Args:
            known_faces_dir (str): Directory containing known face images
        """
        self.face_recognition = FaceRecognition()
        self.known_faces_dir = known_faces_dir
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load all known faces from the directory"""
        known_path = Path(self.known_faces_dir)
        if known_path.exists():
            for person_dir in known_path.iterdir():
                if person_dir.is_dir():
                    self.face_recognition.load_person_faces(person_dir.name, str(person_dir))
                    logger.info(f"Loaded faces for {person_dir.name}")
        else:
            logger.warning(f"Known faces directory '{self.known_faces_dir}' not found")
    
    def identify_from_camera(self):
        """Run real-time face identification from camera"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Cannot open camera")
            return
        
        logger.info("Starting face identification (Press 'q' to quit)")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Identify faces
                identifications = self.face_recognition.identify_faces(rgb_small_frame)
                
                # Draw results on frame
                frame = self.draw_identifications(frame, identifications)
                
                # Display the frame
                cv2.imshow('Face Identification', frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def draw_identifications(self, frame, identifications):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: OpenCV frame
            identifications: List of identification results
        
        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        
        for detection in identifications:
            top, right, bottom, left = detection['face_location']
            name = detection['name']
            confidence = detection['confidence']
            
            # Scale back up face locations
            top = top * 4
            right = right * 4
            bottom = bottom * 4
            left = left * 4
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.6:
                color = (0, 165, 255)  # Orange for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            label = f"{name} ({confidence:.2f})"
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def identify_from_image(self, image_path):
        """
        Identify faces in a single image
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            List of identifications
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Cannot read image: {image_path}")
            return []
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        identifications = self.face_recognition.identify_faces(rgb_image)
        
        # Draw and display
        annotated = self.draw_identifications(image, identifications)
        cv2.imshow('Face Identification Result', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return identifications

if __name__ == "__main__":
    app = FaceIdentificationApp()
    app.identify_from_camera()
