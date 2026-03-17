import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model("pet_breed_classifier.h5")



import json

with open("class_names.json") as f:
    class_names = json.load(f)

st.title("🐶🐱 Cat & Dog Breed Classifier")

uploaded_file = st.file_uploader("Upload a pet image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    index = np.argmax(prediction)
    confidence = np.max(prediction)*100

    breed = class_names[index]

    if breed[0].isupper():
        animal = "Cat"
    else:
        animal = "Dog"

    st.success(f"Animal: {animal}")
    st.success(f"Breed: {breed}")
    st.write(f"Confidence: {confidence:.2f}%")

    # show top-3 predicted breeds
    top3 = prediction[0].argsort()[-3:][::-1]
    for i in top3:
        st.write(class_names[i], f"{prediction[0][i]*100:.2f}%")