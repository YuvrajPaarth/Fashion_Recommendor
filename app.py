import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from annoy import AnnoyIndex  # Annoy library for faster approximate search
from numpy.linalg import norm

# Load pre-computed feature embeddings and file names
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))  # Feature vectors of images
filenames = pickle.load(open('filenames.pkl', 'rb'))  # List of image file paths

# Define the Annoy index
num_dimensions = feature_list.shape[1]  # Number of dimensions of the feature vectors
annoy_index = AnnoyIndex(num_dimensions, 'angular')  # Using angular distance metric

# Build the Annoy index by adding each feature vector
for i, feature in enumerate(feature_list):
    annoy_index.add_item(i, feature)  # Add each feature vector with its index
annoy_index.build(10)  # Build the index using 10 trees (higher trees = better accuracy but slower build)

# Load the pre-trained ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False  # Freeze the model's weights as we are using it only for inference

# Add a GlobalMaxPooling2D layer to reduce the output of ResNet50 to a 1D feature vector
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Streamlit application title
st.title('Fashion Recommender System')

# Function to save the uploaded file in the 'uploads' folder
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1  # Return success status
    except:
        return 0  # Return failure status

# Function to extract features from an image using the pre-trained ResNet50 model
def feature_extraction(img_path, model):
    # Load the image and resize it to the input size required by the model
    img = image.load_img(img_path, target_size=(224, 224))
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Expand the dimensions to create a batch of size 1
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image for the ResNet50 model
    preprocessed_img = preprocess_input(expanded_img_array)
    # Predict features using the model and flatten the output
    result = model.predict(preprocessed_img).flatten()
    # Normalize the feature vector for consistency
    normalized_result = result / norm(result)
    return normalized_result

# Function to recommend similar images using the Annoy index
def recommend(features, annoy_index, n=5):
    # Find the 'n' nearest neighbors to the input feature vector
    indices, distances = annoy_index.get_nns_by_vector(features, n, include_distances=True)
    return distances, indices  # Return the distances and indices of the nearest neighbors

# Steps for the recommender system
## 1. File upload -> save
uploaded_file = st.file_uploader("Choose an image")  # File uploader widget
if uploaded_file is not None:  # Check if a file has been uploaded
    if save_uploaded_file(uploaded_file):  # Save the uploaded file
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image)  # Show the uploaded image in the app
        
        # Extract features from the uploaded image
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        
        # Get recommendations using the Annoy index
        distances, indices = recommend(features, annoy_index)
        
        # Display the recommended images in columns
        col1, col2, col3, col4, col5 = st.columns(5)  # Create 5 columns for displaying recommendations
        for i, col in enumerate([col1, col2, col3, col4, col5]):  # Loop through each column
            with col:  # Add content to the column
                img = Image.open(filenames[indices[i]])  # Load the recommended image
                resized_img = img.resize((200, 200))  # Resize the image for better display
                # Calculate the confidence score (1 - distance)
                confidence_score = 1 - distances[i]
                # Display the image with its confidence score
                st.image(resized_img, caption=f"Recommendation {i+1} (Score: {confidence_score:.2f})")
    else:
        st.header("Some error occurred in file upload")  # Show an error message if file upload fails