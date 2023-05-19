import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from model import load_images_from_directory, classify_image

# Load the model
model = hub.load("https://kaggle.com/models/rishitdagli/plant-disease/frameworks/TensorFlow2/variations/plant-disease/versions/1")

# Define the image directory
directory = 'path/to/your/image/directory'

# Load images from the directory
images = load_images_from_directory(directory)

# Iterate over the images and classify them
for image in images:
    predicted_label = classify_image(model, image)
    print('Predicted Label:', predicted_label)