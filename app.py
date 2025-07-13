import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import pandas as pd

# Load your model
model = tf.keras.models.load_model('product_classifier.h5')

# Load class names from the model (if available) or from a file
if hasattr(model, 'class_names'):
    class_names = model.class_names
else:
    # Fallback: load from a file (e.g., class_names.txt, one class per line)
    with open('class_names.txt') as f:
        class_names = [line.strip() for line in f]

# Load product prices from CSV
prices_df = pd.read_csv('product_prices.csv')
product_prices = dict(zip(prices_df['product'], prices_df['price']))

# Session state for cart
if 'cart' not in st.session_state:
    st.session_state.cart = []

st.title("🛒 Smart Shopping Cart (Streamlit)")

# Webcam input
st.header("📷 Capture Product Image")
img_file_buffer = st.camera_input("Take a picture of the product")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    st.image(image, caption="Captured Image", use_column_width=True)
    # Preprocess image for model
    img = image.resize((224, 224))  # Change size if your model uses a different input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Predict class
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    class_name = class_names[class_idx]
    price = product_prices.get(class_name, "N/A")
    st.success(f"**Product:** {class_name}")
    st.info(f"**Price:** ${price}")
    # Add to cart button
    if st.button("Add to Cart"):
        st.session_state.cart.append((class_name, price))
        st.success(f"Added {class_name} to cart!")

# Remove last item button
if st.button("Remove Last Item from Cart"):
    if st.session_state.cart:
        removed = st.session_state.cart.pop()
        st.warning(f"Removed {removed[0]} from cart.")
    else:
        st.info("Cart is already empty.")

# Cart display
st.header("🧾 Cart")
if st.session_state.cart:
    for item, price in st.session_state.cart:
        st.write(f"**{item}**  ${price}")
    st.markdown("---")
    st.markdown(f"**Total Items:** {len(st.session_state.cart)}")
    total_bill = sum([price for _, price in st.session_state.cart if isinstance(price, (int, float))])
    st.markdown(f"**Total Bill:** ${total_bill:.2f}")
else:
    st.write("Cart is empty.")
