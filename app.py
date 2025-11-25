import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained CNN model
model = load_model("cnn_image_classifier.keras")

# Class names (for CIFAR-10 example)
class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# Streamlit App UI
# Adds a title and a small description at the top of the app.
st.title("🖼️ CNN Image Classifier")
st.write("Upload an image and let the model predict the class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]) # Restricts uploads to only jpg, jpeg, png image files.
# If the user uploads a file, uploaded_file will contain it; otherwise, it’s None.

if uploaded_file is not None:
    # Open original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image (Original)", use_container_width=True) # Displays the uploaded image in the app with a caption.


    # Preprocess (resize only for model input)
    img_resized = image.resize((32, 32)) # CIFAR-10 uses 32x32 so we will resize the uploaded image. Opens image using Pillow (PIL) and resizes it to 32x32 (since CIFAR-10 expects that size).
    img_array = np.array(img_resized) / 255.0 # Divides by 255.0 to normalize pixel values (CNN models expect inputs between 0 and 1).
    img_array = np.expand_dims(img_array, axis=0) # Adds an extra batch dimension (axis=0) → shape becomes (1, 32, 32, 3) instead of (32, 32, 3).

    # Predict
    predictions = model.predict(img_array) # model.predict(img_array) gives probabilities for all classes.
    predicted_class = class_names[np.argmax(predictions)] # np.argmax(predictions) picks the class with the highest probability.

    st.success(f"Prediction: {predicted_class}")
