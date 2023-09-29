# Tamagoyaki
ML projects

Task Summary:
The code implements an image search engine using a pre-trained ResNet50 model for feature extraction. The main components include:

1.Vectorization: Images are transformed into feature vectors using the ResNet50 model.
2.Similarity Search: Given a query image, the system finds the most similar images from the database based on cosine similarity.
3.Visualization: The results are visualized, showing the top similar images along with their similarity scores.
4.Testing: The code includes a testing script using the unittest framework to ensure the correctness of key functionalities.

LIBRARIES

# Import necessary libraries
import numpy as np
from typing import Sequence
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from typing import List, Tuple
import matplotlib.pyplot as plt
import unittest
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load pre-trained ResNet50 model for image feature extraction
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Define a function to preprocess an image
def preprocess_image(img_path):
    # Load the image and convert it into a format suitable for ResNet50
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

  return img_array
  
VECTORIZER

# Create a class for image vectorization using ResNet50
class Vectorizer:
    def __init__(self):
        # Initialize ResNet50 model for image vectorization
        self.resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Transform a sequence of images into feature vectors
  def transform(self, images: Sequence[str]) -> np.ndarray:
      features = []
      for img_path in images:
            # Preprocess each image and extract features using ResNet50
          preprocess_img = self._preprocess_image(img_path)
          if preprocess_img is not None:
              feature = self.resnet_model.predict(preprocess_img)
              flattened_feature = feature.flatten()
              features.append(flattened_feature)

        return np.array(features)

    # Internal method to preprocess an image
  def _preprocess_image(self, img_path):
      try:
            # Try to load and preprocess the image
          img = image.load_img(img_path, target_size=(224, 224))
          img_array = image.img_to_array(img)
          img_array = np.expand_dims(img_array, axis=0)
          img_array = preprocess_input(img_array)
          return img_array
       except Exception as e:
            # If an error occurs during image processing, print an error message and return None
          print(f"Error processing image at {img_path}: {e}")
          return None
          
SEARCH ENGINE

# Create a class for an image search engine
class ImageSearchEngine:
    def __init__(self, vectorizer, image_vectors):
        # Initialize the search engine with a vectorizer and precomputed image vectors
        self.vectorizer = vectorizer
        self.image_vectors = image_vectors

    # Find the most similar images to a query image
  def most_similar(self, query: np.ndarray, n: int = 5) -> List[Tuple[float, str]]:
        # Transform the query image into a feature vector
      query_vector = self.vectorizer.transform([query])
        # Calculate cosine similarities between the query vector and the database vectors
      similarities = cosine_similarity(query_vector, self.image_vectors)[0]
        # Get the indices of the top similar images
      top_indices = np.argsort(similarities)[::-1][:n]

        # Visualize the top similar images
  plt.figure(figsize=(15, 3))
      for i, index in enumerate(top_indices):
          similarity = similarities[index]
          image_name = f"Image_{index}"
          plt.subplot(1, n, i + 1)
          plt.imshow(plt.imread(f"C:/Users/Flepkica/Desktop/pictures/image_db/{image_name}.jpg"))
          plt.title(f"Similarity: {similarity:.2f}\nImage Name: {image_name}")
      plt.axis('off')
  plt.show()

        # Return a list of tuples containing similarity scores and image names
  return [(similarities[index], f"Image_{index}") for index in top_indices]

    # Visualize a list of similar images
  def visualize_similar_images(self, similar_images: List[Tuple[float, str]]) -> None:
      plt.figure(figsize=(15, 3))
      for i, (similarity, image_name) in enumerate(similar_images):
          plt.subplot(1, len(similar_images), i + 1)
          img_path = f"C:/Users/Flepkica/Desktop/pictures/image_db/{image_name}.jpg"
          img = plt.imread(img_path)
          plt.imshow(img)
          plt.title(f"Similarity: {similarity:.2f}\nImage Name: {image_name}")
          plt.axis('off')
      plt.show()

# Path to the image database
image_db_path = 'C:/Users/Flepkica/Desktop/pictures/image_db'

# Get a list of image paths in the database
image_paths = [os.path.join(image_db_path, file_name) for file_name in os.listdir(image_db_path) if file_name.endswith(('.jpg', '.jpeg', '.png'))]

# Create a vectorizer and transform images into feature vectors
vectorizer = Vectorizer()
image_vectors = vectorizer.transform(image_paths)

# Create an image search engine
search_engine = ImageSearchEngine(vectorizer, image_vectors)

# Path to the query image
query_image_paths = ['C:/Users/Flepkica/Desktop/pictures/test/flower.jpg']

# Transform the query image into a feature vector
query_image = vectorizer.transform(query_image_paths)

# Find and print the most similar images to the query image
similar_images = search_engine.most_similar(query_image, n=5)
for similarity, image_name in similar_images:
    print(f"Similarity: {similarity}, Image Name: {image_name}")

TEST
# Create a test class for the image search engine
class TestImageSearchEngine(unittest.TestCase):

    # Set up the test environment
  def setUp(self):
      image_db_path = 'C:/Users/Flepkica/Desktop/pictures/image_db'
      image_paths = [os.path.join(image_db_path, file_name) for file_name in os.listdir(image_db_path) if file_name.endswith(('.jpg', '.jpeg', '.png'))]
      vectorizer = Vectorizer()
      self.image_vectors = vectorizer.transform(image_paths)
      self.search_engine = ImageSearchEngine(vectorizer, self.image_vectors)

    # Test the most_similar method
  def test_most_similar(self):
      query_image_path = 'C:/Users/Flepkica/Desktop/pictures/test/flower.jpg'
      query_image = self.search_engine.vectorizer.transform([query_image_path])
      similar_images = self.search_engine.most_similar(query_image, n=5)

        # Assertion: Check if the result is a list
  self.assertIsInstance(similar_images, list)

        # Assertion: Check if the list is not empty
  self.assertGreater(len(similar_images), 0, "List of similar images should not be empty")

    # Test the visualize_similar_images method
  def test_visualize_similar_images(self):
      similar_images = [(0.9, 'Image_1'), (0.85, 'Image_2'), (0.8, 'Image_3')]
      self.assertIsNone(self.search_engine.visualize_similar_images(similar_images))

# Run the tests if the script is executed directly
if __name__ == '__main__':
    unittest.main()

