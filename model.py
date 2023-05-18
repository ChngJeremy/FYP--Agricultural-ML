import os
import numpy as np
import cv2
from plantcv import plantcv as pcv
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# Set the path to the dataset directory
data_dir = 'path/to/dataset'

# Set the image size for resizing
image_size = (224, 224)

# Set the batch size and number of epochs
batch_size = 32
epochs = 10

# Create separate directories for training and testing images
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Get the plant class names from the subdirectories in the training directory
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)

# Create an image data generator for data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Create the training data generator
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Create the validation data generator
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Create a callback to save the model weights after each epoch
checkpoint_callback = ModelCheckpoint(
    'model_weights.h5',
    monitor='val_loss',
    save_weights_only=True,
    save_best_only=False,
    verbose=1
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[checkpoint_callback]
)

# Evaluate the model on the testing set
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

# Generate predictions on the testing set
test_generator.reset()
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
predicted_labels = np.argmax(predictions, axis=1)

# Get the true labels from the test generator
true_labels = test_generator.classes

# Get the class indices and their corresponding labels
class_indices = test_generator.class_indices
labels = [class_name for class_name in class_indices.keys()]

# Generate a classification report and confusion matrix
report = classification_report(true_labels, predicted_labels, target_names=labels)
matrix = confusion_matrix(true_labels, predicted_labels)
print(report)
print(matrix)

# Perform image processing and feature extraction on the test set using PlantCV
test_features = []
for image_path in test_generator.filenames:
    # Load the image
    img = cv2.imread(os.path.join(test_dir, image_path))

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

# Train a Support Vector Machine (SVM) classifier using the extracted features
clf = SVC()
clf.fit(features, y_train)

# Use the trained classifier to predict the plant condition for the test set
predictions = clf.predict(test_features)

# Calculate the accuracy of the classifier
accuracy = np.mean(predictions == true_labels)
print('Plant Condition Accuracy:', accuracy)
