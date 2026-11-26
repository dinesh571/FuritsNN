import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="FruitVision AI",
    page_icon="🍎",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------

st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b);
}

.main-title{
    text-align:center;
    color:white;
    font-size:3rem;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:#cbd5e1;
    margin-bottom:25px;
}

.card{
    padding:20px;
    border-radius:16px;
    background:#111827;
    color:white;
    box-shadow:0px 4px 20px rgba(0,0,0,.3);
}

.footer{
    text-align:center;
    color:#94a3b8;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fruit_classifier.keras")

# -------------------------------
# LOAD CLASSES
# -------------------------------

@st.cache_data
def load_classes():
    with open("classes.json", "r") as f:
        return json.load(f)

try:
    model = load_model()
    class_names = load_classes()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------------
# HEADER
# -------------------------------

st.markdown(
    "<div class='main-title'>🍎 FruitVision AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Deep Learning Fruit Recognition System</div>",
    unsafe_allow_html=True
)

# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.title("📊 Model Information")

st.sidebar.info("""
Dataset: Fruits-360
Classes: 278
Framework: TensorFlow
Frontend: Streamlit
Image Size: 100x100
""")

# -------------------------------
# IMAGE UPLOAD
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload Fruit Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    img = image.resize((100, 100))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[predicted_index] * 100)

    st.subheader("📈 Training Accuracy Graph")

    st.image("accuracy_graph.png")

    st.subheader("📉 Training Loss Graph")

    st.image("loss_graph.png")

    with col2:

        st.markdown(f"""
        <div class='card'>
            <h2>Prediction</h2>
            <h1>{predicted_class}</h1>
            <h3>{confidence:.2f}% Confidence</h3>
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence / 100)

    st.subheader("🏆 Top 5 Predictions")

    top5 = np.argsort(prediction)[-5:][::-1]

    for idx in top5:
        st.write(
            f"{class_names[idx]} : {prediction[idx] * 100:.2f}%"
        )

st.markdown("---")

st.markdown(
    "<div class='footer'>Built with TensorFlow + Streamlit</div>",
    unsafe_allow_html=True
)