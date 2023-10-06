# Tamagoyaki
ML projects

Task Summary:
The code implements an image search engine using a pre-trained ResNet50 model for feature extraction. The main components include:

1.Vectorization: Images are transformed into feature vectors using the ResNet50 model.


2.Similarity Search: Given a query image, the system finds the most similar images from the database based on cosine similarity.


3.Visualization: The results are visualized, showing the top similar images along with their similarity scores.


4.Testing: The code includes a testing script using the unittest framework to ensure the correctness of key functionalities.


LIBRARIES



     import os
     import numpy as np
     from typing import Sequence, List, Tuple
     from keras.preprocessing import image
     from keras.applications import ResNet50
     from keras.applications.resnet50 import preprocess_input
     from sklearn.metrics.pairwise import cosine_similarity
     import matplotlib.pyplot as plt
     from PIL import Image
     
     
     class Vectorizer:
         def __init__(self):
             self.resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
     
         def _preprocess_image(self, img_path):
             try:
                 img = image.load_img(img_path, target_size=(224, 224))
                 img_array = image.img_to_array(img)
                 img_array = np.expand_dims(img_array, axis=0)
                 img_array = preprocess_input(img_array)
                 return img_array
             except Exception as e:
                 print(f"Error processing image at {img_path}: {e}")
                 return None
     
         def transform(self, images: Sequence[str]) -> np.ndarray:
             features = []
             for img_path in images:
                 preprocess_img = self._preprocess_image(img_path)
                 if preprocess_img is not None:
                     feature = self.resnet_model.predict(preprocess_img)
                     flattened_feature = feature.flatten()
                     features.append(flattened_feature)
     
             return np.array(features)
     
     class ImageSearchEngine:
         def __init__(self, vectorizer, image_vectors):
             self.vectorizer = vectorizer
             self.image_vectors = image_vectors
             self.query_vector = None
     
         def most_similar(self, query: np.ndarray, n: int = 5) -> List[Tuple[float, str]]:
             self.query_vector = self.vectorizer.transform(query)
             similarities = cosine_similarity(self.query_vector, self.image_vectors)
     
             top_indices = np.argsort(similarities)[:, ::-1][:, :n]
     
             flat_indices = top_indices.flatten()
     
             similar_images = [(similarities[i, j], f"Image_{j}") for i, j in enumerate(flat_indices)]
     
             return similar_images
     
         def visualize_similar_images(self, similar_images: List[Tuple[float, str]], image_db_path: str, query_image_path: str) -> None:
             plt.figure(figsize=(15, 3))
             plt.subplot(1, len(similar_images) + 1, 1)
             query_img = plt.imread(query_image_path)
             plt.imshow(query_img)
             plt.title("Query Image")
             plt.axis('off')
     
             for i, (similarity, image_name) in enumerate(similar_images):
                 plt.subplot(1, len(similar_images) + 1, i + 2)
                 img_path = os.path.join(image_db_path, f"{image_name}.jpg")
                 img = plt.imread(img_path)
                 plt.imshow(img)
                 plt.title(f"Similarity: {similarity:.2f}\nImage Name: {image_name}")
                 plt.axis('off')
     
             plt.show()
     
         def create_collage(self, similar_images: List[Tuple[float, str]], image_db_path: str, output_path: str) -> None:
             if not similar_images:
                 print("No similar images found.")
                 return
     
             images = []
             for _, image_name in similar_images:
                 img_path = os.path.join(image_db_path, f"{image_name}.jpg")
                 img = Image.open(img_path)
                 images.append(img)
     
             width, height = images[0].size
             collage_width = width * min(len(images), 3)
             collage_height = height * ((len(images) - 1) // 3 + 1)
     
             collage = Image.new('RGB', (collage_width, collage_height))
     
             for i, img in enumerate(images):
                 collage.paste(img, (i % 3 * width, i // 3 * height))
     
             collage.save(output_path)
     
     image_db_path = 'C:/Users/Flepkica/Desktop/pictures/image_db'
     
     image_paths = [os.path.join(image_db_path, file_name) for file_name in os.listdir(image_db_path) if file_name.endswith(('.jpg', '.jpeg', '.png'))]
     
     vectorizer = Vectorizer()
     image_vectors = vectorizer.transform(image_paths)
     
     search_engine = ImageSearchEngine(vectorizer, image_vectors)
     
     query_image_paths = ['C:/Users/Flepkica/Desktop/pictures/test/pizza.jpg']
     query_image_path = query_image_paths[0]
     
     batch_size = 32
     image_batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

     for batch in image_batches:
         image_vectors_batch = vectorizer.transform(batch)

         similar_images = search_engine.most_similar(image_vectors_batch, n=5)
         search_engine.visualize_similar_images(similar_images, image_db_path, query_image_path)
         search_engine.create_collage(similar_images, image_db_path, 'output_collage.jpg')
 
