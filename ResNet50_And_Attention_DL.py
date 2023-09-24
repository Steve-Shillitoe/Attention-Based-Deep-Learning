import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

def gallery_show(images):
    for i in range(len(images)):
        image = images[i]
        plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(image))
        plt.axis("off")
    plt.show()
    plt.savefig('NoisyImage.jpg')
    

# Load a pre-trained CNN model (e.g., ResNet-50)
cnn_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
cnn_model.trainable = False  # Freeze the pre-trained weights

# Define the attention mechanism
class SelfAttention(layers.Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def build(self, input_shape):
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        batch_size, height, width, channels = x.shape

        # Compute query, key, and value
        query = layers.Conv2D(channels // 8, (1, 1), activation="relu")(x)
        key = layers.Conv2D(channels // 8, (1, 1), activation="relu")(x)
        value = layers.Conv2D(channels, (1, 1))(x)

        # Compute attention scores
        attention = tf.matmul(query, key, transpose_b=True)
        attention = tf.nn.softmax(attention)

        # Compute the attended feature maps
        out = tf.matmul(attention, value)
        
        # Resize the output to match the input shape
        out = tf.image.resize(out, (height, width))

        # Combine with the original input using gamma
        gamma = self.add_weight("gamma", shape=(1,), initializer="zeros", trainable=True)
        out = gamma * out + x
        return out

# Combine the CNN and attention mechanism
class CNNWithAttention(models.Model):
    def __init__(self, cnn_model, attention):
        super(CNNWithAttention, self).__init__()
        self.cnn = cnn_model
        self.attention = attention

    def call(self, x):
        features = self.cnn(x)
        attended_features = self.attention(features)
        return attended_features

# Load an example image and apply preprocessing
image_path = 'Ferret_Images\\Karla_And_Missy.jpg'
img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(img)
original_image = img
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# Create the model with attention
attention_model = CNNWithAttention(cnn_model, SelfAttention())

# Forward pass the image through the model
output = attention_model(img)

# The 'output' now contains the attended features from the image
print(output)

# Assuming 'output' contains the attended image feature map
attended_image = output[0]  # Extract the feature map for the first (and only) image in the batch

# Select a slice (e.g., the first channel) from the feature map
slice_of_feature_map = attended_image[:, :, 0]

# Normalize the slice to the range [0, 1]
min_val = tf.reduce_min(slice_of_feature_map)
max_val = tf.reduce_max(slice_of_feature_map)
normalized_slice = (slice_of_feature_map - min_val) / (max_val - min_val)

plt.imshow(original_image)
plt.title('Original Image')
plt.show()

# Display the slice as a heatmap using Matplotlib
plt.imshow(normalized_slice, cmap='viridis')  # You can choose a different colormap if desired
#plt.axis('off')  # Turn off axis labels
plt.colorbar()  # Add a colorbar to the heatmap
plt.title('Attended Feature Map')
plt.show()

gallery_show([original_image, normalized_slice ])