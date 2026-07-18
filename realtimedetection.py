import os

print("Current Folder:", os.getcwd())
print("XML Exists:", os.path.exists("haarcascade_frontalface_default.xml"))

import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load Model
with open("emotionrecogniser.json", "r") as json_file:
    model_json = json_file.read()
    

model = model_from_json(model_json)
model.load_weights("emotionrecogniser.h5")   # Make sure this file exists

# Load Face Detector
import os

cascade_path = os.path.join(
    os.path.dirname(__file__),
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Could not load:", cascade_path)
    exit()

# Feature Extraction
def extract_features(image):
    image = image.astype("float32") / 255.0
    image = image.reshape(1, 48, 48, 1)
    return image

# Start Webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

while True:
    ret, frame = webcam.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))

        img = extract_features(face)

        prediction = model.predict(img, verbose=0)
        emotion = labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()