import os
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from PIL import Image, UnidentifiedImageError

# Load Pre-trained ResNet50 Model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Add Global Max Pooling layer
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to Extract Features
def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        
        if features.shape == (1, 2048):  # Ensure correct shape
            return features.flatten()
        else:
            print(f"Skipping {img_path} due to unexpected feature shape: {features.shape}")
            return None
    
    except (UnidentifiedImageError, FileNotFoundError) as e:
        print(f"Skipping {img_path}: {str(e)}")
        return None

# Load Image Filenames
image_dir = "images"
filenames = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('jpg', 'png', 'jpeg'))]

# Extract Features
feature_list = []
valid_filenames = []

for file in tqdm(filenames, desc="Extracting Features"):
    features = extract_features(file, model)
    if features is not None:
        feature_list.append(features)
        valid_filenames.append(file)  # Store only valid images

# Convert to Numpy Array & Save
feature_list = np.array(feature_list)
pickle.dump(feature_list, open("embeddings.pkl", "wb"))
pickle.dump(valid_filenames, open("filenames.pkl", "wb"))

print(f"Feature extraction complete! {len(feature_list)} valid images processed.")
