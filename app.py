import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load your trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cloth_condition_model.h5")
    return model

model = load_model()

st.title("👕 Cloth Condition & Value Prediction App")

st.write("Upload an image of a cloth and provide details to predict its condition and estimated resale/recycle value.")

# Image upload
uploaded_file = st.file_uploader("Upload a cloth image", type=["jpg", "jpeg", "png"])

# Manual inputs
brand = st.selectbox("Brand Quality", ["Low", "Medium", "High"])
fabric = st.selectbox("Fabric Type", ["Cotton", "Polyester", "Denim", "Silk", "Other"])
usage = st.slider("Approximate Age of Cloth (in years)", 0, 10, 1)
category = st.selectbox("Clothing Type", ["Shirt", "Pants", "Dress", "Jacket", "Other"])

if uploaded_file is not None:
    # Read and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Cloth", use_container_width=True)

    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    if st.button("Predict Condition"):
        prediction = model.predict(img)
        condition = "Torn / Not Usable" if prediction[0][0] > 0.5 else "Good / Reusable"

        # Example price estimation logic
        base_price = 1000  # base resale value
        if brand == "High":
            base_price *= 1.5
        elif brand == "Low":
            base_price *= 0.7

        if condition == "Torn / Not Usable":
            price = 0
            status = "♻️ Recyclable Only"
        else:
            price = base_price - (usage * 50)
            status = "💸 Sellable"

        st.subheader(f"🧵 Condition: {condition}")
        st.subheader(f"{status}")
        st.subheader(f"💰 Estimated Value: ₹{max(price, 0):.2f}")
