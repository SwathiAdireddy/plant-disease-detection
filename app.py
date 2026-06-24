import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import numpy as np


class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]


@st.cache_resource
def load_model():

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(15, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.load_weights("plant_weights.weights.h5")

    return model


model = load_model()



st.title("🌿 Plant Disease Detection")

st.write(
    "Upload a plant leaf image and the model will predict the disease."
)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )

    
    img = image.resize((224, 224))

    
    img_array = np.array(img)

    
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions)

    confidence = np.max(predictions) * 100

    predicted_class = class_names[predicted_index]

    # Make label more readable
    display_name = predicted_class.replace("___", " - ")
    display_name = display_name.replace("__", " ")
    display_name = display_name.replace("_", " ")

    st.success(
        f"Prediction: {display_name}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    
    # SHOW TOP 3 PREDICTIONS

    st.subheader("Top 3 Predictions")

    top_3_indices = np.argsort(predictions[0])[-3:][::-1]

    for idx in top_3_indices:
        class_name = class_names[idx]
        class_name = class_name.replace("___", " - ")
        class_name = class_name.replace("__", " ")
        class_name = class_name.replace("_", " ")

        prob = predictions[0][idx] * 100

        st.write(f"**{class_name}** : {prob:.2f}%")
