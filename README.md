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

FeatureExtractor class

Responsible for extracting features from images using the VGG16 model. The model is customized to return features from the fully-connected layer named 'fc1'.
The fully-connected layer 'fc1' typically serves as a feature extractor in VGG16.

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
     
     query_image_path = "C:\\Users\\PC\\Desktop\\simple_image_retrieval_dataset\\test\\leopard.jpg"
     database_path = "C:\\Users\\PC\\Desktop\\simple_image_retrieval_dataset\\image_db"
     collage_output_path = "C:\\Users\\PC\\Desktop\\Collage.png"
     
     fe = FeatureExtractor()
     query_feature = fe.extract(img=Image.open(query_image_path))
     
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
     
     features = np.array(features)
Similarity calculation 

Calculation of the Euclidean distance between the features of the query image and database images 

     dists = np.linalg.norm(features - query_feature, axis=1)
     
The indices of the top 6 similar images are obtained, and a list of tuples containing similarity scores and image paths is created

     ids = np.argsort(dists)[:6]
     scores = [(dists[id], paths[id]) for id in ids]
     
Visualization 

     axes = []
     fig = plt.figure(figsize=(8, 8))
     for a in range(2, 8):
         score = scores[a - 2]
         axes.append(fig.add_subplot(2, 3, a - 1))
         subplot_title = str(score[0])
         axes[-1].set_title(subplot_title)
         plt.axis('off')
         try:
             plt.imshow(Image.open(score[1]))
         except Exception as e:
     
             print(f"Error displaying image at {score[1]}: {str(e)}. Skipping.")
     fig.tight_layout()
     
     fig.savefig(collage_output_path)
     
     plt.show()

 
