# Gender & Age Detection

A Python computer vision project to detect faces and predict the gender and age of a person in real-time using OpenCV, Deep Neural Networks (DNN), and a Flask web backend.

## Features
- **Professional Web UI:** A beautiful, responsive interface that works seamlessly in the browser.
- **Camera & Image Upload:** UI supports both uploading image files and snapping live photos directly from your webcam.
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

Due to GitHub's file size limits, the large pre-trained model weights (`.caffemodel` and `.pb` files) are excluded from this repository via `.gitignore`. You must download them and place them in the root directory.

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
Then, open your browser and go to: `http://127.0.0.1:5000`

## Usage (CLI)

If you prefer to run the script via the command line without the web interface:

**Run on Default Webcam:**
```bash
python gad.py
```

**Run on an Image:**
```bash
python gad.py --image path/to/image.jpg
```

## Disclaimer
This AI tool uses computer vision algorithms to estimate demographic properties. Model accuracy heavily depends on image quality. Poor lighting, extreme facial angles, occlusions (like glasses or masks), or multiple faces may result in incorrect predictions or errors. Results should be treated as estimates.

## License
This project is open-source and available under the [MIT License](LICENSE).

---
Made by [Prayatshu Misra](https://github.com/PrayatshuMisra)
