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




      
     


     
Comparison between VGG16 and ResNet50 in Machine Learning


1. Architecture:
   
VGG16:
Type: VGG16 (Visual Geometry Group 16-layer)
Depth: Consists of 16 weight layers, including 13 convolutional layers and 3 fully connected layers.
Layer Configurations: Uses small-size convolutional filters (3x3) with max-pooling layers.
ResNet50:
Type: ResNet50 (Residual Network with 50 layers)
Depth: Comprises 50 weight layers with a unique residual block design.
Layer Configurations: Introduces residual blocks that include skip connections, allowing the network to learn residuals.

2. Vanishing Gradient Problem:
   
VGG16:
Issue: Prone to vanishing gradient problem, especially in deeper networks.
Solution: Uses smaller convolutional filters, but deep networks may still suffer from gradient-related challenges.
ResNet50:
Advantage: Addresses vanishing gradient problem effectively through residual connections.
Solution: Skip connections allow for the direct flow of gradients, enabling the training of very deep networks.

3. Network Learning:
   
VGG16:
Learning Style: Learns hierarchical features layer by layer.
Strength: Effective for relatively shallower networks.
ResNet50:
Learning Style: Learns residual functions, facilitating the learning of identity mappings.
Strength: Particularly effective for very deep networks, enabling the training of networks with hundreds of layers.

4. Parameter Efficiency:

VGG16:
Parameter Count: Generally has more parameters due to a higher number of convolutional filters.
Complexity: Can be computationally expensive, especially for large image datasets.
ResNet50:
Parameter Count: Fewer parameters due to the use of residual blocks.
Efficiency: More parameter-efficient, allowing for the training of deeper networks with fewer resources.

5. Training Speed:
   
VGG16:
Training Time: May require longer training times, especially for deep architectures.
Convergence: Slower convergence, especially for very deep networks.
ResNet50:
Training Time: Faster training times, thanks to the use of residual connections.
Convergence: Faster convergence, making it suitable for large-scale datasets.

6. Performance on Image Recognition:

VGG16:
Recognition Accuracy: Achieves high accuracy on image recognition tasks.
Applications: Commonly used for image classification in various domains.
ResNet50:
Recognition Accuracy: Demonstrates state-of-the-art accuracy, especially on challenging datasets.
Applications: Widely used in computer vision applications, including object detection and image segmentation.

7. Transfer Learning:

VGG16:
Transferability: Pre-trained VGG16 models are effective for transfer learning tasks in various domains.
ResNet50:
Transferability: ResNet50's pre-trained models are highly transferable and perform well in diverse tasks.

8. Memory Usage:

VGG16:
Memory Consumption: Generally requires more memory due to a higher number of parameters.
ResNet50:
Memory Consumption: More memory-efficient, making it suitable for deployment on resource-constrained devices.

9. Common Use Cases:

VGG16:
Use Cases: Effective for image classification, feature extraction, and artistic style transfer.
Applications: Commonly used in research and practical applications.
ResNet50:
Use Cases: Ideal for deep image recognition tasks, object detection, and segmentation.
Applications: Widely applied in industry-standard computer vision solutions.

10. Conclusion:

VGG16:
Advantages: Simplicity, effective for smaller networks, widely used in research.
Considerations: May not scale well for very deep networks.
ResNet50:
Advantages: Residual connections, effective for deep networks, state-of-the-art accuracy.
Considerations: May be overkill for simpler tasks; efficient for large-scale applications.

Conclusion:

The choice between VGG16 and ResNet50 depends on the specific requirements of the task at hand. VGG16 is a robust and simpler architecture suitable for various image recognition tasks. On the other hand, ResNet50 shines in scenarios requiring very deep networks and achieves state-of-the-art results, especially on challenging datasets. Both architectures have made significant contributions to the field of computer vision, and the choice should be guided by the complexity of the problem and available computational resources. Due to its simple architecture, VGG16 is less demanding when it comes to computational resources.


Comparison between Euclidean distance and Cosine similarity in Machine Learning

Cosine Similarity:

Advantages:
Scale-Invariant: Insensitive to the scale of features, making it robust when the magnitude of differences is not crucial.
Directional Emphasis: Emphasizes the orientation or direction of feature vectors, capturing similarity based on angles.

Definition:
Cosine similarity measures the cosine of the angle between two non-zero vectors. In the context of image similarity, it assesses the similarity of direction between feature vectors.

Euclidean Distance:

Definition:
Euclidean distance, a classic metric, measures the straight-line distance between two points in Euclidean space. In the context of image similarity, it computes the geometric distance between feature vectors representing images.

Advantages:
Sensitive to Scale: Takes into account the magnitude of feature differences, making it effective when scale is critical.
Intuitive Interpretation: Geometrically interpretable, providing an intuitive understanding of similarity in feature space.

6 images with the most similarity value correspondent to the query:

a) Localy : 

![collage](https://github.com/Platipipus645/Tamagoyaki/assets/76967479/659b29e4-524f-420b-9bcc-1bb5a8527294)






