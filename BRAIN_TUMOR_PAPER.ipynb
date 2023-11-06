import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image

# Function to load and preprocess MRI scans
def load_and_preprocess_data(module_path, label, num_samples):
    data = []
    labels = []
    for i in range(num_samples):
        filename = f"{module_path}/y{i}.jpg" if label == 1 else f"{module_path}/no{i}.jpg"
        img = Image.open(filename)
        img = img.resize((150, 150))  # Resize images to a common size
        img_array = np.asarray(img)  # Convert image to array
        data.append(img_array)
        labels.append(label)
    return data, labels

# Load and preprocess data from "yes" and "no" modules
num_samples = 1500  # Number of samples in each class
yes_data, yes_labels = load_and_preprocess_data("/content/drive/MyDrive/Colab Notebooks/mri/yes", 1, num_samples)
no_data, no_labels = load_and_preprocess_data("/content/drive/MyDrive/Colab Notebooks/mri/no", 0, num_samples)

# Ensure all images have consistent shape (150x150x3)
print("Reshaping and checking dimensions for yes_data:")
yes_data = [img.reshape(150, 150, 3) for img in yes_data if img.shape == (150, 150, 3)]
yes_labels = [label for img, label in zip(yes_data, yes_labels) if img.shape == (150, 150, 3)]

print("Reshaping and checking dimensions for no_data:")
no_data = [img.reshape(150, 150, 3) for img in no_data if img.shape == (150, 150, 3)]
no_labels = [label for img, label in zip(no_data, no_labels) if img.shape == (150, 150, 3)]

# Combine data and labels for MRI dataset
X_mri = np.array(yes_data + no_data)
y_mri = np.array(yes_labels + no_labels)

# Split the MRI data into training and testing sets
X_mri_train, X_mri_test, y_mri_train, y_mri_test = train_test_split(X_mri, y_mri, test_size=0.2, random_state=42)

# Convert MRI labels to categorical format
y_mri_train_categorical = to_categorical(y_mri_train, num_classes=2)

# Build and train the MRI CNN model
mri_model = Sequential()
mri_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
mri_model.add(MaxPooling2D((2, 2)))
mri_model.add(Flatten())
mri_model.add(Dense(64, activation='relu'))
mri_model.add(Dense(2, activation='softmax'))
mri_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mri_history = mri_model.fit(X_mri_train, y_mri_train_categorical, epochs=10, batch_size=32, validation_split=0.1)

# Plotting accuracy per epoch for MRI model
plt.plot(mri_history.history['accuracy'], label='MRI Training Accuracy')
plt.plot(mri_history.history['val_accuracy'], label='MRI Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
