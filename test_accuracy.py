import json
import matplotlib.pyplot as plt

# Load the epochs list from the JSON file
with open('epochs.json', 'r') as f:
    epochs_list = json.load(f)

# Load the history dictionary from the JSON file
with open('history.json', 'r') as f:
    history = json.load(f)

# Replace 'train_accuracy' and 'val_accuracy' with the actual accuracy values
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plot the accuracy curve
plt.plot(epochs_list, train_accuracy, 'b.-', label='Training Accuracy')
plt.plot(epochs_list, val_accuracy, 'r.-', label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()