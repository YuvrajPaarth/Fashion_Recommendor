import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2 
import matplotlib.pyplot as plt


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))


model = ResNet50(weights='imagenet', include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img('sample/jersey.jpg',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])

print(indices)

for file in indices[0]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
    print("Press any key to see the next image or 'q' to quit.")
    key = cv2.waitKey(0)
    if key == ord('q'):  # Exit loop if 'q' is pressed
        break
    cv2.destroyAllWindows()  # Close the current window
cv2.destroyAllWindows()  # Ensure all windows are closed at the end

# Display all recommended images in a grid
# plt.figure(figsize=(15, 5))  # Adjust figure size

# for i, file in enumerate(indices[0]):
#     temp_img = cv2.imread(filenames[file])
#     temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
#     plt.subplot(1, 5, i + 1)  # Create a subplot for each image
#     plt.imshow(temp_img)
#     plt.title(f"Rec {i+1}")
#     plt.axis('off')

# plt.tight_layout()
# plt.show()