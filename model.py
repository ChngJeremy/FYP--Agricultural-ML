import os
import numpy as np
import cv2
from plantcv import plantcv as pcv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Set the path to the directory containing the plant images
data_dir = 'path/to/dataset'

# Set the paths to the image directories for different plant conditions
healthy_dir = os.path.join(data_dir, 'healthy')
unhealthy_dir = os.path.join(data_dir, 'unhealthy')

# Set up empty lists to store image paths and labels
image_paths = []
labels = []

# Load the healthy plant images
for filename in os.listdir(healthy_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(healthy_dir, filename)
        image_paths.append(image_path)
        labels.append(0)  # Label 0 for healthy plants

# Load the unhealthy plant images
for filename in os.listdir(unhealthy_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(unhealthy_dir, filename)
        image_paths.append(image_path)
        labels.append(1)  # Label 1 for unhealthy plants

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Perform image processing and feature extraction on the training set
features = []
for image_path in X_train:
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply PlantCV preprocessing and analysis steps
    # Modify this section to suit your specific analysis needs
    # Example steps: segmentation, feature extraction
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = np.zeros_like(img_binary)
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    features.append(pcv.measurements.shannon_entropy(mask))

# Train a Support Vector Machine (SVM) classifier using the extracted features
clf = SVC()
clf.fit(features, y_train)

# Perform image processing and feature extraction on the test set
test_features = []
for image_path in X_test:
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply PlantCV preprocessing and analysis steps
    # Modify this section to suit your specific analysis needs
    # Example steps: segmentation, feature extraction
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = np.zeros_like(img_binary)
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    test_features.append(pcv.measurements.shannon_entropy(mask))
    
# Use the trained classifier to predict the plant condition for the test set
predictions = clf.predict(test_features)

# Calculate the accuracy of the classifier
accuracy = np.mean(predictions == y_test)
print('Accuracy: {:.2f}'.format(accuracy))

# Save the classifier for later use
model_path = 'path/to/model'
pcv.save_object(obj=clf, filename=model_path)

