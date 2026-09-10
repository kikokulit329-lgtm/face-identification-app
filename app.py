from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import base64
import numpy as np
from main import FaceIdentificationApp
import logging
from io import BytesIO
import json

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize face identification app
face_app = FaceIdentificationApp()

@app.route('/', methods=['GET'])
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about loaded faces"""
    stats = face_app.face_recognition.get_stats()
    return jsonify(stats)

@app.route('/api/identify', methods=['POST'])
def identify_faces():
    """
    Identify faces in uploaded image
    
    Expected: JSON with 'image' key containing base64 encoded image
    """
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Identify faces
        identifications = face_app.face_recognition.identify_faces(rgb_image)
        
        # Convert face locations to serializable format
        results = []
        for ident in identifications:
            results.append({
                'name': ident['name'],
                'confidence': float(ident['confidence']),
                'location': {
                    'top': int(ident['face_location'][0]),
                    'right': int(ident['face_location'][1]),
                    'bottom': int(ident['face_location'][2]),
                    'left': int(ident['face_location'][3])
                }
            })
        
        return jsonify({
            'success': True,
            'identifications': results
        })
    
    except Exception as e:
        logger.error(f"Error identifying faces: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/add-person', methods=['POST'])
def add_person():
    """
    Add a new person's face to the database
    
    Expected: FormData with 'name' and 'image' file
    """
    try:
        name = request.form.get('name')
        image_file = request.files.get('image')
        
        if not name or not image_file:
            return jsonify({'error': 'Name and image are required'}), 400
        
        # Read image
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Convert to RGB and add
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_app.face_recognition.add_face_encoding(name, rgb_image)
        
        return jsonify({
            'success': True,
            'message': f'Face added for {name}'
        })
    
    except Exception as e:
        logger.error(f"Error adding person: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
