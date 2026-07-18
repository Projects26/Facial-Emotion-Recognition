import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image
from pathlib import Path

# ---------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Facial Emotion Recognition")
st.write("Upload an image to detect facial emotion.")

# ---------------------------------------------------
# Project Directory
# ---------------------------------------------------
BASE_DIR = Path(__file__).parent

# ---------------------------------------------------
# Load Emotion Model
# ---------------------------------------------------
@st.cache_resource
def load_emotion_model():
    try:
        with open(BASE_DIR / "emotiondetector.json", "r") as json_file:
            model_json = json_file.read()

        model = model_from_json(model_json)
        model.load_weights(str(BASE_DIR / "emotiondetector.h5"))

        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_emotion_model()

# ---------------------------------------------------
# Load Haar Cascade
# ---------------------------------------------------
cascade_path = BASE_DIR / "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(str(cascade_path))

if face_cascade.empty():
    st.error("Could not load haarcascade_frontalface_default.xml")
    st.stop()

# ---------------------------------------------------
# Emotion Labels
# ---------------------------------------------------
labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

# ---------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------
def extract_features(image):
    image = image.astype("float32") / 255.0
    image = image.reshape(1, 48, 48, 1)
    return image

# ---------------------------------------------------
# Upload Image
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        st.warning("No face detected in the uploaded image.")

    else:

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))

            img = extract_features(face)

            prediction = model.predict(img, verbose=0)

            emotion = labels[np.argmax(prediction)]

            cv2.rectangle(
                image_np,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image_np,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        st.success("Emotion detected successfully!")

    st.image(
        image_np,
        channels="RGB",
        use_container_width=True
    )