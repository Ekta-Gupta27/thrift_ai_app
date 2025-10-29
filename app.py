import streamlit as st
import numpy as np
import cv2
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---- Load Model ----
model = load_model("cloth_condition_model.h5")
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ---- App Title ----
st.title("♻️ Smart Thrift Store – AI Sustainability Predictor")

st.write("""
Upload your clothing image to analyze its condition,  
predict whether it should be **Recycled** or **Resold**,  
and estimate its **resale price and environmental savings** 🌍
""")

# ---- Image Upload ----
uploaded_file = st.file_uploader("Upload clothing image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess for the category model
    img = image.load_img(uploaded_file, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    pred_class = class_labels[np.argmax(pred)]

    st.subheader(f"🧥 Predicted Clothing Type: {pred_class}")

    # ---- Auto Image Condition Analysis ----
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        img_cv = cv2.imread(tmp_file.name)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / edges.size

    auto_torn = "Yes" if edge_density > 0.12 or brightness < 70 else "No"

    # ---- Manual Inputs ----
    st.subheader("👕 Additional Condition Details")
    age = st.number_input("Age of the cloth (in years):", min_value=0, max_value=20, step=1)
    faded = st.selectbox("Is it faded?", ["No", "Yes"])
    branded = st.selectbox("Is it branded?", ["Yes", "No"])
    user_torn = st.selectbox("Is it torn?", ["Auto Detect", "Yes", "No"])

    # Decide final torn value
    if user_torn == "Auto Detect":
        torn = auto_torn
    else:
        torn = user_torn

    st.write(f"🩹 Torn or Damaged (AI + User): **{torn}**")

    # ---- Decision Logic ----
    if torn == "Yes" or faded == "Yes" or age > 3:
        decision = "♻️ Recyclable"
    else:
        decision = "🛍️ Sellable"

    # ---- Price Prediction ----
    base_price = 500 if branded == "Yes" else 200
    depreciation = max(0.1, 1 - (age * 0.15))
    if decision == "Recyclable":
        price = base_price * 0.4 * depreciation
    else:
        price = base_price * depreciation

    st.subheader(f"✅ Recommended Action: {decision}")
    st.success(f"💰 Estimated Resale Price: ₹{price:.2f}")

    # ---- Sustainability Info ----
    if decision == "Sellable":
        st.info("🌿 Buying pre-loved saves ~2700L of water & 5kg CO₂ per item.")
    else:
        st.info("🔄 Recycling helps divert ~1kg of textile waste from landfills.")
