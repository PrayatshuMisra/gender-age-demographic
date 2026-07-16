# Gender and Age Detection

A Python computer vision script to detect faces and predict the gender and age of a person in real-time using OpenCV and Deep Neural Networks (DNN).

## Features
- **Face Detection:** Uses a Single Shot Detector (SSD) framework with a ResNet base network to accurately locate faces in the frame.
- **Age Prediction:** Classifies age into 8 distinct ranges: `(0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)`.
- **Gender Prediction:** Classifies gender as either `Male` or `Female`.
- **Image or Webcam Input:** Supports analyzing both static images and live video feeds from your webcam.

## Prerequisites

- Python 3.x
- OpenCV (`opencv-python`)

You can install the required dependency via pip:
```bash
pip install opencv-python
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

## Usage

### Run on Webcam
Simply run the script without any arguments to use your default webcam:
```bash
python gad.py
```

### Run on an Image
Pass the path to an image using the `--image` argument:
```bash
python gad.py --image man1.jpg
```

*(Press any key to close the window when viewing a static image).*

## How it Works
1. The script first processes the input frame to detect a face using a pre-trained SSD face detector.
2. If a face is found, it extracts a bounding box, applies padding, and crops the region of interest (ROI).
3. The cropped face ROI is passed sequentially into the `Gender` and `Age` classification Caffe models.
4. The prediction results are then drawn onto the original image as an overlay.
