# Tamagoyaki
ML projects

Task Summary:
The code implements an image search engine using a pre-trained ResNet50 model for feature extraction. The main components include:

1.Vectorization: Images are transformed into feature vectors using the ResNet50 model.


2.Similarity Search: Given a query image, the system finds the most similar images from the database based on cosine similarity.


3.Visualization: The results are visualized, showing the top similar images along with their similarity scores.


4.Testing: The code includes a testing script using the unittest framework to ensure the correctness of key functionalities.


LIBRARIES 

Importing necessary libraries, including tools for working with images (PIL), deep learning model (VGG16) and NumPy.

     import os
     import numpy as np
     import matplotlib.pyplot as plt
     from keras.preprocessing import image
     from keras.applications.vgg16 import VGG16, preprocess_input
     from keras.models import Model
     from pathlib import Path
     from PIL import Image
     from sklearn.metrics.pairwise import cosine_similarity

     class FeatureExtractor:
         def __init__(self):
             base_model = VGG16(weights='imagenet')
             self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
     
         def extract(self, img):
             img = img.resize((224, 224))
             img = img.convert('RGB')
             x = image.img_to_array(img)
             x = np.expand_dims(x, axis=0)
             x = preprocess_input(x)
             feature = self.model.predict(x)[0]
             return feature / np.linalg.norm(feature)
     
     def cosine_similarity(query_image_vector, image_vectors):
         dot_products = np.dot(image_vectors, query_image_vector)
         query_norm = np.linalg.norm(query_image_vector)
         image_norms = np.linalg.norm(image_vectors, axis=1)
         similarities = dot_products / (query_norm * image_norms)
         return similarities
     
     query_image_path = "/home/senji/Desktop/Kvisko/ML/simple_image_retrieval_dataset/test-cases/cat.jpg"
     database_path = "/home/senji/Desktop/Kvisko/ML/simple_image_retrieval_dataset/image-db"
     collage_output_path = "/home/senji/Desktop/Kvisko/ML/collage.jpg"
     features_save_path = "/home/senji/Desktop/Kvisko/ML/features.npz"
     
     
     if os.path.exists(features_save_path):
         saved_features = np.load(features_save_path)
         features, paths = saved_features['features'], saved_features['paths']
     else:
         fe = FeatureExtractor()
     
         features = []
         paths = []
     
         for img_path in sorted(os.listdir(database_path)):
             try:
                 if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                     feature = fe.extract(img=Image.open(os.path.join(database_path, img_path)))
                     features.append(feature)
                     paths.append(os.path.join(database_path, img_path))
             except Exception as e:
                 print(f"Error processing image at {img_path}: {str(e)}. Skipping.")
                 continue
     
         features = np.array(features)
         
         np.savez(features_save_path, features=features, paths=paths)
     
     query_feature = fe.extract(img=Image.open(query_image_path))
     
     similarities = cosine_similarity(query_feature, features)
     
     num_similar_images = min(5, len(similarities))
     ids = np.argsort(similarities)[::-1][:num_similar_images]
     scores = [(similarities[id], paths[id]) for id in ids]
     
     axes = []
     fig = plt.figure(figsize=(10, 10))
     
     axes.append(fig.add_subplot(3, 2, 1))
     axes[-1].set_title("Query Image")
     plt.axis('off')
     plt.imshow(Image.open(query_image_path))
     
     for a in range(2, num_similar_images + 2):
         score = scores[a - 2]
         axes.append(fig.add_subplot(3, 2, a))
         subplot_title = f"Similarity: {score[0]:.2f}"
         axes[-1].set_title(subplot_title)
         plt.axis('off')
         try:
             plt.imshow(Image.open(score[1]))
         except Exception as e:
             print(f"Error displaying image at {score[1]}: {str(e)}. Skipping.")
     
     fig.tight_layout()
     
     fig.savefig(collage_output_path)
     
     plt.show()




      
     


     


6 images with the most similarity value correspondent to the query:

a) Localy : 

![collage](https://github.com/Platipipus645/Tamagoyaki/assets/76967479/659b29e4-524f-420b-9bcc-1bb5a8527294)






