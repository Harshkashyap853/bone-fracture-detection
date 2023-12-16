import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV3Large



def create_model():
    pretrained_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling=None)

    pretrained_model.trainable = True

    x = pretrained_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    
    
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=pretrained_model.input, outputs=outputs)
    return model

def trainPart(part):
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    image_dir = THIS_FOLDER + '/Dataset/'
    data = load_path(image_dir, part)
    labels = []
    filepaths = []

    for row in data:
        labels.append(row['label'])
        filepaths.append(row['image_path'])

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    images = pd.concat([filepaths, labels], axis=1)

    train_df, test_df = train_test_split(images, train_size=0.9, shuffle=True, random_state=1)

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
        validation_split=0.2
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input
    )

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
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
        class_mode='categorical',
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
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    model = create_model()
    print("-------Training " + part + "-------")

    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7)
    ]

    history = model.fit(train_images, validation_data=val_images, epochs=50, callbacks=callbacks, batch_size=32)

    model.save(os.path.join(image_dir, "weights", "MobileNetV3_" + part + "_frac.h5"))
    results = model.evaluate(test_images, verbose=0)
    print(part + " Results:")
    print(results)
    print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Run the function for each part in the array
categories_parts = [ "Hand"]
for category in categories_parts:
    trainPart(category)
