# Face Recognition System 🎭

A Python-based face recognition system that can identify known faces in images using facial encoding and comparison algorithms.

## 📋 Description

This Face Recognition System allows you to:
- Load and encode faces from a directory of known individuals
- Process unknown images to identify faces
- Draw bounding boxes and labels on detected faces
- Save results with annotated images

The system uses the `face_recognition` library (built on dlib) to perform accurate facial recognition with customizable tolerance levels.

## 🚀 Features

- **Face Detection**: Automatically detects faces in images
- **Face Recognition**: Compares unknown faces against a database of known faces
- **Visual Output**: Draws green bounding boxes and labels on recognized faces
- **Batch Processing**: Process multiple unknown images at once
- **Customizable Tolerance**: Adjust recognition sensitivity
- **Error Handling**:  Robust error handling for file processing

## 📁 Project Structure

```
Face-Recognition/
├── face.py                 # Main face recognition system
├── known_faces/           # Directory for known faces (organized by person name)
│   ├── person1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── person2/
│       └── image1.jpg
└── unknown_faces/         # Directory for images to be processed
    ├── test1. jpg
    └── test2.jpg
```

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Install Dependencies

```bash
pip install face-recognition opencv-python numpy
```

**Note**: The `face_recognition` library requires `dlib`, which may need additional system dependencies: 

**On Ubuntu/Debian:**
```bash
sudo apt-get install cmake libboost-all-dev
```

**On macOS:**
```bash
brew install cmake boost
```

**On Windows:**
- Install CMake from [cmake.org](https://cmake.org/download/)
- Or use pre-built wheels:  `pip install dlib`

## 💻 Usage

### 1. Prepare Your Data

Create the directory structure: 

```bash
mkdir -p known_faces unknown_faces
```

Add known faces: 
- Create subdirectories in `known_faces/` for each person
- Add one or more photos of each person in their respective folder
- Use clear, front-facing photos for best results

### 2. Run the System

```python
from face import FaceRecognitionSystem

# Initialize the system
system = FaceRecognitionSystem(
    known_faces_dir="known_faces",
    unknown_faces_dir="unknown_faces",
    tolerance=0.6,  # Lower = more strict, Higher = more lenient
    model="hog"     # 'hog' for CPU, 'cnn' for GPU (more accurate)
)

# Load known faces
system.load_known_faces()

# Process unknown faces
system.process_unknown_faces()
```

### 3. View Results

Output images will be saved with the prefix `output_` in the current directory, showing:
- Green rectangles around detected faces
- Names of recognized individuals

## ⚙️ Configuration

### Parameters

- **tolerance** (float, default:  0.6): 
  - Face comparison tolerance
  - Lower values = stricter matching (more false negatives)
  - Higher values = looser matching (more false positives)
  - Recommended range: 0.4 - 0.6

- **model** (str, default: "hog"):
  - `"hog"`: Faster, less accurate, CPU-based
  - `"cnn"`: Slower, more accurate, requires GPU

## 📊 Example Output

When processing images, you'll see console output like:

```
Loading known faces...
Loading faces for John Doe...
  Processing photo1.jpg
  Successfully loaded face from photo1.jpg

Processing unknown faces...
Processing group_photo.jpg... 
Loading image...
Image loaded successfully. Shape: (1080, 1920, 3)
Finding faces...
Found 3 faces
Found match: John Doe
Result saved to output_group_photo.jpg
```

## 🔍 How It Works

1. **Encoding Known Faces**: The system loads images from `known_faces/`, detects faces, and creates 128-dimensional encodings for each face
2. **Processing Unknown Images**: For each image in `unknown_faces/`, it:
   - Detects all faces
   - Creates encodings for detected faces
   - Compares encodings with known faces
   - Draws annotations on matches
3. **Output**:  Saves annotated images showing recognized faces

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with [face_recognition](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- Uses [dlib](http://dlib.net/) for facial recognition algorithms
- OpenCV for image processing

## 📧 Contact

**Author**: kamalstores  
**GitHub**: [@kamalstores](https://github.com/kamalstores)

---

⭐ If you find this project useful, please consider giving it a star! 
```
