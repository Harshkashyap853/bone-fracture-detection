import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV3Large

# Load only "Hand" data
def load_hand_data(path):
    dataset = []
    for folder in os.listdir(path):
        folder = os.path.join(path, folder)
        if os.path.isdir(folder):
            for body in os.listdir(folder):
                if body == "Hand":  # Include only "Hand" category
                    path_p = os.path.join(folder, body)
                    for id_p in os.listdir(path_p):
                        patient_id = id_p
                        path_id = os.path.join(path_p, id_p)
                        for lab in os.listdir(path_id):
                            if lab.split('_')[-1] == 'positive':
                                label = 'fractured'
                            elif lab.split('_')[-1] == 'negative':
                                label = 'normal'
                            path_l = os.path.join(path_id, lab)
                            for img in os.listdir(path_l):
                                img_path = os.path.join(path_l, img)
                                dataset.append({
                                    'label': label,
                                    'image_path': img_path
                                })
    return dataset

# Load only "Hand" data from path
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(THIS_FOLDER, 'Dataset')
data = load_hand_data(image_dir)

# Update Labels
Labels = ["fractured", "normal"]

# Convert to DataFrame
labels = [row['label'] for row in data]
filepaths = [row['image_path'] for row in data]

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

images = pd.concat([filepaths, labels], axis=1)

# Split dataset: 10% test, 90% train (90% train will split to 20% validation and 80% train)
train_df, test_df = train_test_split(images, train_size=0.9, shuffle=True, random_state=1)

# ImageDataGenerators
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
    validation_split=0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input
)

# Flow from DataFrame
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='binary',  # Binary classification
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='binary',
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=False
)

# Model
pretrained_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
pretrained_model.trainable = False

inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(50, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Binary classification
model = tf.keras.Model(inputs, outputs)
print(model.summary())

# Model Compilation
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Training
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_images, validation_data=val_images, epochs=25, callbacks=[callbacks])

# Save Model
model.save(os.path.join(THIS_FOLDER, "weights", "MobileNetV3_Hand.h5"))

# Evaluate on test set
results = model.evaluate(test_images, verbose=0)
print(results)
print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")

# Plot Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
