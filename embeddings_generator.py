import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import os
import pickle
from tqdm import tqdm
from PIL import UnidentifiedImageError

# Load the pre-trained ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Add a GlobalMaxPooling2D layer to reduce the output of ResNet50 to a 1D feature vector
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to extract features from an image
def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))  # Load image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess for ResNet50
        
        features = model.predict(img_array, verbose=0)  # Extract features
        return features.flatten()  # Flatten into 1D vector
    
    except (UnidentifiedImageError, OSError) as e:
        print(f"❌ Skipping {img_path} due to error: {e}")
        return None  # Skip corrupted images

# Define the image folder
image_folder = 'images'  # Ensure this folder exists with dataset

# Get all image filenames
if not os.path.exists(image_folder):
    print(f"❌ Error: The folder '{image_folder}' does not exist!")
    exit()

filenames = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.lower().endswith(('png', 'jpg', 'jpeg'))]

if len(filenames) == 0:
    print("❌ Error: No valid image files found in the folder!")
    exit()

# Extract features from each image
feature_list = []
valid_filenames = []

for file in tqdm(filenames, desc="🔍 Extracting Features"):
    features = extract_features(file, model)
    if features is not None:
        feature_list.append(features)
        valid_filenames.append(file)  # Store only valid filenames

# Convert to NumPy array to ensure consistency
feature_list = np.array(feature_list, dtype=np.float32)

# Save extracted features and valid filenames
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(valid_filenames, open('filenames.pkl', 'wb'))

print("\n✅ Feature extraction completed successfully! Embeddings saved.")