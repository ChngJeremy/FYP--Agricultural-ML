import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from preprocessing import create_data_generators
from model import build_model, compile_model, train_model, save_model_weights
from evaluation import evaluate_model, generate_classification_report
from plantcv_processing import process_image

# Set the path to the dataset directory
data_dir = 'path/to/dataset'

# Set the image size for resizing
image_size = (224, 224)

# Set the batch size and number of epochs
batch_size = 32
epochs = 10

# Set the path to save the model weights
model_weights_path = 'path/to/save/model/weights.h5'

# Create separate directories for training and testing images
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Get the plant class names from the subdirectories in the training directory
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)

# Create data generators for training, validation, and testing
train_generator, val_generator, test_generator = create_data_generators(train_dir, val_dir, test_dir, image_size, batch_size)

# Build the model architecture
model = build_model(image_size, num_classes)

# Compile the model
compile_model(model)

# Train the model
train_model(model, train_generator, val_generator, epochs)

# Save the model weights
save_model_weights(model, model_weights_path)

# Evaluate the model on the testing data
evaluate_model(model, test_generator)

# Generate classification report and confusion matrix
generate_classification_report(model, test_generator, class_names)

# Perform PlantCV processing on the test images
test_features = []
test_labels = []

for image_path, label in test_generator.filenames, test_generator.classes:
    # Load the image
    image = load_image(image_path)

    # Process the image using PlantCV
    features = process_image(image)

    # Append the features and label to the lists
    test_features.append(features)
    test_labels.append(label)

# Convert the lists to NumPy arrays
test_features = np.array(test_features)
test_labels = np.array(test_labels)

# Train a Support Vector Machine (SVM) classifier using the extracted features
clf = SVC()
clf.fit(test_features, test_labels)

# Use the trained classifier to predict the plant condition for the test set
predictions = clf.predict(test_features)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(test_labels, predictions)
print('Plant Condition Accuracy:', accuracy)
