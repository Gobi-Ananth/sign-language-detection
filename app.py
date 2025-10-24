import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(page_title="Sign Language Alphabet Detector", page_icon="🤟", layout="wide")

st.title("🤟 Sign Language Alphabet Detector (A–Z excluding J and Z)")
st.write("Upload an image or use your webcam to recognize American Sign Language alphabets. \
          The model supports 24 letters (A–Z except J and Z).")

# ----------------------------
# Download model from Google Drive using gdown
# ----------------------------
@st.cache_resource
def download_and_load_model():
    file_id = "1HRzpUqhN83Y5NGumR_HvBao2_XfTqZbo"
    url = f"https://drive.google.com/uc?id={file_id}"
    model_path = "model.h5"

    if not os.path.exists(model_path):
        st.info("📥 Downloading model from Google Drive... please wait ⏳")
        gdown.download(url, model_path, quiet=False)
        st.success("✅ Model downloaded successfully!")

    # Load model
    model = tf.keras.models.load_model(model_path)
    return model

model = download_and_load_model()

# ----------------------------
# Define class names (A–Z excluding J and Z)
# ----------------------------
CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

# ----------------------------
# Prediction function
# ----------------------------
def predict(img):
    # Convert to grayscale (dataset is grayscale)
    img = img.convert('L')
    # Resize to 28x28 (Sign Language MNIST input size)
    img = img.resize((28, 28))
    img_array = np.array(img)

    # Normalize and reshape
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

# ----------------------------
# Sidebar options
# ----------------------------
option = st.sidebar.radio("Choose input source", ("📸 Webcam", "🖼 Upload Image"))

# ----------------------------
# Webcam Capture
# ----------------------------
if option == "📸 Webcam":
    st.write("Click below to capture your sign image.")
    camera = st.camera_input("Capture Sign Image")

    if camera:
        img = Image.open(camera)
        st.image(img, caption="Captured Image", use_container_width=True)
        label, conf = predict(img)
        st.success(f"**Predicted Letter:** {label}  |  **Confidence:** {conf:.2f}%")

# ----------------------------
# Image Upload
# ----------------------------
elif option == "🖼 Upload Image":
    uploaded_file = st.file_uploader("Upload a hand sign image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        label, conf = predict(img)
        st.success(f"**Predicted Letter:** {label}  |  **Confidence:** {conf:.2f}%")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit & TensorFlow")
