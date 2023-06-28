from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def evaluate_model(model, test_generator):
    test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)

def generate_classification_report(model, test_generator, class_names):
    test_generator.reset()
    predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = test_generator.classes
    report = classification_report(true_labels, predicted_labels, target_names=class_names)
    matrix = confusion_matrix(true_labels, predicted_labels)
    print(report)
    print(matrix)
