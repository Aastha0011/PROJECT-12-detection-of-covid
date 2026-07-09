import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="COVID-19 Detection", page_icon="🩺")

st.title("🩺 COVID-19 Detection from Chest X-Ray")
st.write("Upload a Chest X-Ray image to predict whether it is COVID-19 or Normal.")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# ----------------------------
# Upload Image
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose a Chest X-Ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = image.resize((299, 299))

    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]

    st.subheader("Prediction")

    if prediction >= 0.5:
        st.error("🦠 COVID-19 Detected")
        st.write(f"Confidence : **{prediction*100:.2f}%**")

    else:
        st.success("✅ Normal")
        st.write(f"Confidence : **{(1-prediction)*100:.2f}%**")
