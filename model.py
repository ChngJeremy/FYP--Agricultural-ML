import tensorflow_hub as hub
import tensorflow as tf
import numpy as np


def build_model(input_shape, num_classes):
    model = hub.load("https://kaggle.com/models/rishitdagli/plant-disease/frameworks/TensorFlow2/variations/plant-disease/versions/1")
    
    # Adjust the input layer to match the desired input shape
    model.build([None, *input_shape])
    
    # Add a new output layer for the specified number of classes
    model = tf.keras.Sequential([
        model,
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

def preprocess_image(image):
    # Resize the image to match the input shape of the model
    image = image.resize((224, 224))
    
    # Convert the image to a NumPy array
    image_array = np.asarray(image)
    
    # Convert the NumPy array to a float32 data type
    image_array = image_array.astype('float32')
    
    # Normalize the pixel values from [0, 255] to [-1, 1]
    normalized_image_array = (image_array / 127.0) - 1
    
    # Return the preprocessed image as a NumPy array
    return normalized_image_array

def create_data_generator(preprocess_function, batch_size, image_size, shuffle=False):
    return tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_function,
        batch_size=batch_size,
        target_size=image_size,
        shuffle=shuffle
    )

def load_dataset(dataset_path, image_size, batch_size):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )
    
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(  
        dataset_path,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )
    
    return train_dataset, val_dataset

def image_dataset_from_directory(directory, labels='inferred', label_mode='int', class_names=None, color_mode='rgb', batch_size=32, image_size=(256, 256), shuffle=True, seed=None, validation_split=None, subset=None, interpolation='bilinear', follow_links=False):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels=labels,
        label_mode=label_mode,
        class_names=class_names,
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        interpolation=interpolation,
        follow_links=follow_links
    )

def load_image(image_path): 
    return tf.keras.preprocessing.image.load_img(image_path)


def train_model(model, train_generator, val_generator, epochs):
    model.fit(train_generator, validation_data=val_generator, epochs=epochs)

def save_model_weights(model, weights_path):
    model.save_weights(weights_path)

def load_model_weights(model, weights_path):
    model.load_weights(weights_path)

def classify_image(model, image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Reshape the preprocessed image to match the expected input shape
    input_data = np.expand_dims(preprocessed_image, axis=0)
    
    # Make predictions using the model
    predictions = model.predict(input_data)
    
    # Get the predicted class label
    predicted_label = np.argmax(predictions, axis=1)
    
    return predicted_label