from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np

app = Flask(__name__)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

faceProto = os.path.join(BASE_DIR, "opencv_face_detector.pbtxt")
faceModel = os.path.join(BASE_DIR, "opencv_face_detector_uint8.pb")
ageProto = os.path.join(BASE_DIR, "age_deploy.prototxt")
ageModel = os.path.join(BASE_DIR, "age_net.caffemodel")
genderProto = os.path.join(BASE_DIR, "gender_deploy.prototxt")
genderModel = os.path.join(BASE_DIR, "gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

print("Loading AI Models into memory...")
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
print("Models loaded successfully!")

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    # Keep swapRB=False to ensure BGR order is preserved for Caffe models
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
    
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            
    return faceBoxes

@app.route('/')
def index():
    # Serve our static HTML frontend
    return send_file(os.path.join(BASE_DIR, 'index.html'))

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
        
    try:
        # Read the uploaded image from memory without saving to disk
        in_memory_file = file.read()
        nparr = np.frombuffer(in_memory_file, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
             return jsonify({'error': 'Invalid image format'}), 400

        # 1. Detect face
        faceBoxes = getFaceBox(faceNet, frame)
        if not faceBoxes:
            return jsonify({'error': 'No face detected in the image.'}), 404

        # We will process the first face found in the image
        faceBox = faceBoxes[0]
        padding = 20
        face = frame[max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)]

        # 2. Pre-process cropped face for age & gender networks
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # 3. Predict Gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        # 4. Predict Age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        
        return jsonify({
            'gender': gender, 
            'age': f'{age[1:-1]} years'
        })
        
    except Exception as e:
        return jsonify({'error': 'An internal server error occurred: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
