import streamlit as st
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import json
from PIL import Image
import webbrowser


st.set_page_config(page_title="🐶 Dog Breed Classifier", page_icon="🐕", layout="centered")

st.markdown("""
    <h1 style='text-align:center;'>🐶 Dog Breed Classification</h1>
    <p style='text-align:center;'>Upload a dog image and I’ll predict its breed with AI magic ✨</p>
""", unsafe_allow_html=True)


try:
    with open("breed_info.json", "r") as f:
        breed_info = json.load(f)
except:
    breed_info = {}


@st.cache_resource
def load_detector():
    return hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1")

detector = load_detector()


@st.cache_resource
def load_breed_model():
    return tf.keras.models.load_model("dog_breed_model.h5", custom_objects={"KerasLayer": hub.KerasLayer})

breed_model = load_breed_model()

try:
    with open("class_labels.txt", "r") as f:
        class_labels = [line.strip() for line in f.readlines()]
except:
    class_labels = ["beagle", "labrador", "bulldog", "pug", "dalmatian", "golden_retriever"]


uploaded_file = st.file_uploader("📸 Upload Dog Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    st.info("🔍 Checking if the image contains a dog...")

    
    image_np = np.array(image) / 255.0
    input_tensor = tf.expand_dims(image_np, 0)

    try:
        detections = detector(input_tensor)
        detection_classes = [cls.decode("utf-8") for cls in detections["detection_class_entities"][0].numpy()]
        dog_detected = any("Dog" in cls for cls in detection_classes)
    except Exception:
        st.warning("⚠️ Detector could not verify image type. Proceeding with breed prediction anyway.")
        dog_detected = True  

    if not dog_detected:
        st.error("🚫 Please upload a **dog image**! 🐕")
        st.stop()


    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    with st.spinner("⏳ Analyzing image... please wait"):
        preds = breed_model.predict(img_batch)

    breed_idx = np.argmax(preds)
    conf = np.max(preds) * 100

    if breed_idx < len(class_labels):
        breed_name = class_labels[breed_idx]
    else:
        breed_name = f"Unknown Breed (index {breed_idx})"


    if conf < 30:
        st.error(f"⚠️ Confidence only **{conf:.2f}%** — please upload a clearer dog image.")
        st.stop()

    st.success(f"🐕 Predicted Breed: **{breed_name.replace('_', ' ').title()}**")
    st.info(f"💡 Confidence: **{conf:.2f}%**")

    top3_idx = np.argsort(preds[0])[-3:][::-1]
    st.markdown("### 🔝 Top 3 Predictions:")
    for i in top3_idx:
        if i < len(class_labels):
            st.write(f"- {class_labels[i].replace('_', ' ').title()} ({preds[0][i]*100:.2f}%)")


    st.markdown("### 🐾 Breed Information")
    info = breed_info.get(breed_name.lower())

    if info:
        st.write(f"**Origin:** {info.get('origin', 'Unknown')}")
        st.write(f"**Size:** {info.get('size', 'Unknown')}")
        st.write(f"**Temperament:** {info.get('temperament', 'Unknown')}")
    else:
        st.warning("ℹ️ Sorry, I don’t have detailed info about this breed yet.  PLEASE CLICK BELOW")

    wiki_url = f"https://en.wikipedia.org/wiki/{breed_name.replace('_', ' ')}"
    if st.button("📚 Know More About This Breed"):
        webbrowser.open_new_tab(wiki_url)
        st.markdown(f"[Click here if not opening automatically →]({wiki_url})")

st.markdown("<hr><p style='text-align:center;'>Made with ❤️ using TensorFlow & Streamlit</p>", unsafe_allow_html=True)
