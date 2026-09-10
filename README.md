# 🔍 Face Identification App

A real-time face identification application using computer vision and deep learning to identify people on camera or in photos.

---

## 📋 Quick Access Guide

### 🚀 Installation (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/kikokulit329-lgtm/face-identification-app.git
cd face-identification-app

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create known faces directory
mkdir known_faces
```

### 🎯 Quick Start (Choose One)

**Option A: Web Interface (Recommended)**
```bash
python app.py
```
Then open your browser to: **http://localhost:5000**

**Option B: Command Line (Real-time Camera)**
```bash
python main.py
```
Press 'q' to quit

---

## 📁 Project Structure

```
face-identification-app/
├── main.py                 # CLI application
├── app.py                  # Flask web server
├── face_recognition_module.py  # Core recognition engine
├── requirements.txt        # Dependencies
├── templates/
│   └── index.html         # Web interface
├── known_faces/           # Store face images here
│   ├── PersonA/
│   │   ├── photo1.jpg
│   │   └── photo2.jpg
│   └── PersonB/
│       └── photo1.jpg
└── README.md
```

---

## 🎓 How to Use

### 1. **Add People to Database**

Create folders in `known_faces/` directory:

```
known_faces/
├── John/
│   ├── john1.jpg
│   ├── john2.jpg
│   └── john3.jpg
├── Sarah/
│   ├── sarah1.jpg
│   └── sarah2.jpg
└── Mike/
    ├── mike1.jpg
    ├── mike2.jpg
    └── mike3.jpg
```

**Tips for best results:**
- Use 3-5 clear photos per person
- Include different lighting conditions
- Include different angles (front, side profile)
- Ensure faces are clearly visible
- Use JPG, PNG, or BMP formats

### 2. **Run the Application**

#### Web Interface
```bash
python app.py
```

**Features:**
- 📸 Upload images to identify faces
- ➕ Add new people via web form
- 📊 View database statistics
- 🎨 Modern responsive UI

#### Command Line
```bash
python main.py
```

**Features:**
- 🎥 Real-time camera feed
- 👤 Live face identification
- 🟢 Green box = High confidence
- 🟠 Orange box = Medium confidence
- 🔴 Red box = Low confidence

---

## 🔧 API Endpoints (Web Interface)

### Get Statistics
```http
GET /api/stats
```

**Response:**
```json
{
  "total_encodings": 15,
  "unique_people": 3,
  "people": ["John", "Sarah", "Mike"],
  "model": "hog",
  "tolerance": 0.6
}
```

### Identify Faces in Image
```http
POST /api/identify
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,..."
}
```

**Response:**
```json
{
  "success": true,
  "identifications": [
    {
      "name": "John",
      "confidence": 0.95,
      "location": {
        "top": 100,
        "right": 200,
        "bottom": 300,
        "left": 150
      }
    }
  ]
}
```

### Add New Person
```http
POST /api/add-person
Content-Type: multipart/form-data

name: "John"
image: <binary image data>
```

**Response:**
```json
{
  "success": true,
  "message": "Face added for John"
}
```

---

## ⚙️ Configuration

### Change Face Recognition Model

Edit `app.py` or `main.py`:

```python
# HOG Model (Faster, CPU-friendly)
face_app = FaceIdentificationApp(model='hog')

# CNN Model (Slower, more accurate, requires CUDA)
face_app = FaceIdentificationApp(model='cnn')
```

### Adjust Tolerance (Sensitivity)

```python
# Lower = Stricter matching (fewer false positives)
face_app.face_recognition.tolerance = 0.5

# Higher = Looser matching (more matches)
face_app.face_recognition.tolerance = 0.7
```

---

## 🆘 Troubleshooting

### Issue: "No faces detected"
- ✅ Ensure images are well-lit and clear
- ✅ Face should be at least 10-20 pixels in size
- ✅ Try switching model from 'hog' to 'cnn'

### Issue: "Module not found" errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue: Camera not working
```bash
# Try with a different camera index
# Edit main.py and change:
cap = cv2.VideoCapture(0)  # Try 1, 2, etc.
```

### Issue: Low accuracy
- ✅ Add more photos per person (3-5+)
- ✅ Use different angles and lighting
- ✅ Ensure clear, face-focused photos
- ✅ Lower the tolerance value

---

## 📊 Performance Tips

| Setting | Effect |
|---------|--------|
| Model: HOG | ⚡ Fast (CPU), less accurate |
| Model: CNN | 🚀 Slow (GPU required), more accurate |
| Tolerance: 0.5 | 🔒 Strict, fewer false positives |
| Tolerance: 0.7 | 🔓 Loose, more matches |
| Lower resolution | ⚡ Faster processing |

---

## 🔐 Security & Privacy

⚠️ **Important Notes:**
- This application stores face encodings locally
- Ensure proper data protection measures
- Obtain consent before storing face data
- Add authentication for production use
- Use HTTPS when deploying online
- Comply with local face recognition regulations (GDPR, etc.)

---

## 📚 Features

✨ **Real-time Face Detection & Identification**
- Live camera feed processing
- Multi-face detection in single frame
- Confidence scoring for each match

🎯 **Accuracy & Confidence Scoring**
- Returns confidence scores (0-1)
- Visual confidence indicators
- Adjustable tolerance for sensitivity

👥 **Easy Database Management**
- Add people via web interface
- Store multiple encodings per person
- Quick statistics dashboard

🚀 **Web Interface**
- Modern, responsive design
- Drag-and-drop image upload
- Real-time results display
- Mobile-friendly

---

## 📦 Dependencies

- **OpenCV** - Image processing and camera handling
- **face-recognition** - Deep learning-based face recognition
- **NumPy** - Numerical computations
- **Flask** - Web framework
- **Scikit-learn** - Machine learning utilities

---

## 🚀 Next Steps

1. ✅ Clone the repository
2. ✅ Install dependencies
3. ✅ Add face images to `known_faces/` directory
4. ✅ Run `python app.py`
5. ✅ Open http://localhost:5000

---

## 📝 Future Enhancements

- [ ] Database backend for face storage
- [ ] User authentication system
- [ ] Batch processing support
- [ ] Mobile app integration
- [ ] Advanced filtering and search
- [ ] Audit logging
- [ ] Multi-user support
- [ ] Face verification (1:1 matching)
- [ ] API key authentication
- [ ] Docker support

---

## 📄 License

This project is open source and available under the MIT License.

---

## ⚖️ Disclaimer

This application is for educational and authorized use only. Ensure compliance with local laws and regulations regarding face recognition and data privacy. Users are responsible for obtaining proper consent before identifying or storing face data of any individuals.

---

## 💬 Support

For issues, questions, or suggestions:
1. Check the **Troubleshooting** section above
2. Open an issue on GitHub
3. Check existing issues for solutions

---

**Made with ❤️ for face recognition enthusiasts**
