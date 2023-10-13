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

FeatureExtractor CLASS

Responsible for extracting features from images using the VGG16 model. The model is customized to return features from the fully-connected layer named 'fc1'.
The fully-connected layer 'fc1' typically serves as a feature extractor in VGG16.

** In a neural network, a fully connected layer, also known as a dense layer, is a type of layer where each neuron or node in the layer is connected to every neuron in the previous layer. It's called "fully connected" because each neuron in the current layer is linked to every neuron in the preceding layer. This type of connectivity allows the layer to capture complex relationships in the data.

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
     
     query_image_path = "C:\\Users\\PC\\Desktop\\simple_image_retrieval_dataset\\test\\cat.jpg"
     database_path = "C:\\Users\\PC\\Desktop\\simple_image_retrieval_dataset\\image_db"
     collage_output_path = "C:\\Users\\PC\\Desktop\\Collage.png"

     fe = FeatureExtractor()

     query_feature = fe.extract(img=Image.open(query_image_path))

     features = []
     paths = []
     img_names = []

     
          for img_path in sorted(os.listdir(database_path)):
         try:
             if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                 feature = fe.extract(img=Image.open(os.path.join(database_path, img_path)))
                 features.append(feature)
                 img_name = Path(img_path).name
                 img_names.append(img_name)
                 paths.append(os.path.join(database_path, img_path))
         except Exception as e:
             print(f"Error processing image at {img_path}: {str(e)}. Skipping.")

     np.savez("image_data.npz", features=features, img_names=img_names, paths=paths)

     features = np.array(features)
     
SIMILARITY CALCULATION

Calculation of the Euclidean distance between the features of the query image and database images 

     dists = np.linalg.norm(features - query_feature, axis=1)
     
The indices of the top 6 similar images are obtained, and a list of tuples containing similarity scores and image paths is created

     ids = np.argsort(dists)[:6]
     scores = [(dists[id], paths[id]) for id in ids]
     
VISUALIZATION

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
     
Pinecone 

     pinecone.init(api_key='969830ae-6f81-47ea-aea6-7156df76e898', environment='gcp-starter')
     # np_vectors = np.load...__annotations__
     # img_names = pd.read....
     
     index = pinecone.Index('image14')
     vectors = [np_vector.tolist() for np_vector in np_vectors]
     # 'id':'vec1', 
     #   'values':[0.1, 0.2, 0.3, 0.4], 
     
     pairs = [(img_names[i], vector) for i, vector in enumerate(vectors)]
     
     query_results = index.query(vector=pairs[0], top_k=6)
     for match in query_results.to_dict()["matches"]:
         print(match["id"])
     
     saved_data = np.load("image_data.npz")
     np_vectors = saved_data['features']
     img_names = saved_data['img_names']
     paths = saved_data['paths']
     
     vectors = [feature.tolist() for feature in features]
     
     pairs = list(zip(img_names, vectors))
     index.upsert(pairs, batch_size=50)
     
     query_results = index.query(vector=pairs[0][1], top_k=6)
     
     for match in query_results.to_dict()["matches"]:
         print(match["id"])
     
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

![1](https://github.com/Platipipus645/Tamagoyaki/assets/76967479/3ae71021-9e8f-4af3-8dc3-8cefac18712f)

![2](https://github.com/Platipipus645/Tamagoyaki/assets/76967479/a01721b0-c3cf-4d22-8a31-1f53ab09d6a4)

![Figure_1](https://github.com/Platipipus645/Tamagoyaki/assets/76967479/f12ab9f1-1140-4779-8aea-38e337e7fca6)

