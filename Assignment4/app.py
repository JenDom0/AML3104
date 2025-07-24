# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('mnist_cnn.h5')

model = load_model()

st.title("MNIST Digit Classifier")

uploaded = st.file_uploader("Upload a 28×28 grayscale digit (png/jpg)", type=["png","jpg","jpeg"])
if uploaded:
    # preprocess
    img = Image.open(uploaded).convert('L').resize((28,28))
    img_arr = np.array(img).astype("float32")/255.0
    img_arr = img_arr.reshape(1,28,28,1)

    st.image(img, caption="Input Image", use_column_width=False)
    # predict
    pred = np.argmax(model.predict(img_arr), axis=1)[0]
    st.markdown(f"**Predicted digit:** {pred}")
