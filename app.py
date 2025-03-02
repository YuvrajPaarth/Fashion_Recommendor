import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from annoy import AnnoyIndex  # Faster approximate search
from PIL import Image

# Load precomputed feature embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load pre-trained model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Build Annoy index for fast searching
num_dimensions = feature_list.shape[1]
annoy_index = AnnoyIndex(num_dimensions, 'angular')

for i, feature in enumerate(feature_list):
    annoy_index.add_item(i, feature)
annoy_index.build(10)

# Streamlit UI
st.title('Fashion Recommender System')

# Function to extract features from uploaded image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Function to get recommendations
def recommend(features, annoy_index, n=5):
    indices, distances = annoy_index.get_nns_by_vector(features, n, include_distances=True)
    return distances, indices

# File upload
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Extract features from uploaded image
    features = extract_features(uploaded_file, model)

    # Get recommendations
    distances, indices = recommend(features, annoy_index)

    # Display recommended images
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            img = Image.open(filenames[indices[i]])
            st.image(img, caption=f"Score: {1 - distances[i]:.2f}")
