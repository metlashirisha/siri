import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown

# Download model from Google Drive
url = "https://drive.google.com/file/d/1CSG3JOtbo11VYKpPPCJCTVlVkNNMUMcz/view?usp=drive_link"  # Replace YOUR_FILE_ID with actual ID
output = "model_file_30epochs.h5"
if not tf.io.gfile.exists(output):
    gdown.download(url, output, quiet=False)

# Load the trained model with caching
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_file_30epochs.h5')
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

model = load_model()

# Define class labels (6 emotions)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']

# Preprocess the image
def preprocess_image(image):
    img = cv2.resize(image, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
st.title("Facial Emotion Recognition")
st.write("Upload an image to detect the emotion!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    emotion = class_labels[np.argmax(prediction)]

    st.write(f"Predicted Emotion: **{emotion}**")