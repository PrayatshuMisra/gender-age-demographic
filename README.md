# Gender-Age Detection

A Python computer vision project that detects faces and predicts the demographic properties (gender and age) of a person in real-time using OpenCV, Deep Neural Networks (DNN), and a Flask web backend.

## About This Project: AI Engineering & Integration

In the AI industry, there is a distinction between training machine learning models from scratch and engineering real-world applications around existing state-of-the-art models. **This project focuses entirely on AI Engineering and software integration.**

- **What was built in this project:** 
  - The **Computer Vision Pipeline**: The Python code that takes raw image/webcam streams, uses detection models to mathematically locate facial coordinates, surgically crops the region of interest (ROI), and pipes it sequentially through classification models.
  - The **Flask API Backend** that handles secure, in-memory image processing without saving files to disk.
  - The **Professional Web UI** featuring drag-and-drop file uploads, live WebRTC camera capture, and responsive demographic data cards.

- **What uses Pre-Trained Models:**
  - The underlying neural networks (`.caffemodel` and `.pb` weights) used for Age classification, Gender classification, and SSD Face Detection are **pre-trained models**. They were originally trained by computer vision researchers (such as Gil Levi and Tal Hassner) using thousands of images. This project utilizes their pre-trained weights to perform rapid, real-time inference.

## Features
- **Professional Web UI:** A beautiful, responsive interface that works seamlessly in the browser.
- **Camera & Image Upload:** UI supports both uploading image files and snapping live photos directly from your webcam.
- **Privacy First:** Images are processed strictly in-memory and are never saved or stored on the server.
- **Face Detection:** Uses a Single Shot Detector (SSD) framework with a ResNet base network to accurately locate faces.
- **Age Prediction:** Classifies age into 8 distinct ranges: `(0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)`.
- **Gender Prediction:** Classifies gender as either `Male` or `Female`.

## Prerequisites

- Python 3.x
- Flask (`flask`)
- OpenCV (`opencv-python`)

You can install the required dependencies via pip:
```bash
pip install opencv-python flask
```

## Model Weights (Required)

Due to GitHub's file size limits, the large pre-trained model weights (`.caffemodel` and `.pb` files) are excluded from this repository via `.gitignore`. You must download them and place them in the root directory alongside the python scripts.

You will need the following files:
- **Face Detector:**
  - `opencv_face_detector_uint8.pb`
  - `opencv_face_detector.pbtxt` (Already included)
- **Age Detector:**
  - `age_net.caffemodel`
  - `age_deploy.prototxt` (Already included)
- **Gender Detector:**
  - `gender_net.caffemodel`
  - `gender_deploy.prototxt` (Already included)

## Usage (Web App)

To start the local web server with the full graphical interface:
```bash
python app.py
```
Then, open your browser and navigate to: `http://127.0.0.1:5000`

## Usage (CLI)

If you prefer to run the script via the command line without the web interface:

**Run on Default Webcam:**
```bash
python gad.py
```

**Run on a Static Image:**
```bash
python gad.py --image path/to/image.jpg
```

## Disclaimer
This AI tool uses computer vision algorithms to estimate demographic properties. Model accuracy heavily depends on image quality. Poor lighting, extreme facial angles, occlusions (like glasses or masks), or multiple faces may result in incorrect predictions or errors. Results should always be treated strictly as estimates.

---
Made by [Prayatshu Misra](https://github.com/PrayatshuMisra)
