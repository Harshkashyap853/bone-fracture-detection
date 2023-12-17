import numpy as np
import tensorflow as tf
from keras.preprocessing import image

# load the models when import "predictions.py"
model_elbow_frac = tf.keras.models.load_model("weights/MobileNetV3_Elbow_frac.h5")
model_hand_frac = tf.keras.models.load_model("weights/MobileNetV3_Hand_frac.h5")
model_shoulder_frac = tf.keras.models.load_model("weights/MobileNetV3_Shoulder_frac.h5")
model_parts = tf.keras.models.load_model("weights/MobileNetV3_BodyParts.h5")

# categories for each result by index

#   0-Elbow     1-Hand      2-Shoulder
categories_parts = ["Elbow","Hand","Shoulder"]

#   0-fractured     1-normal
categories_fracture = ['fractured', 'normal']


# get image and model name, the default model is "Parts"
# Parts - bone type predict model of 3 classes
# otherwise - fracture predict for each part
def predict(img, model="Parts"):
    size = 224
    if model == 'Parts':
        chosen_model = model_parts
    else:
        if model == 'Elbow':
            chosen_model = model_elbow_frac
        elif model == 'Hand':
            chosen_model = model_hand_frac
        elif model == 'Shoulder':
            chosen_model = model_shoulder_frac

    # load image with 224px224p (the training model image size, rgb)
    temp_img = image.load_img(img, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = np.argmax(chosen_model.predict(images), axis=1)

    # chose the category and get the string prediction
    if model == 'Parts':
        prediction_str = categories_parts[prediction.item()]
    else:
        prediction_str = categories_fracture[prediction.item()]

    return prediction_str



# #new1
# import numpy as np
# import tensorflow as tf
# from keras.preprocessing import image

# # Load the models when importing "predictions.py"
# model_hand_frac = tf.keras.models.load_model("weights/MobileNetV3_Hand_frac.h5")
# model_parts = tf.keras.models.load_model("weights/MobileNetV3_BodyParts.h5")

# # Categories for each result by index
# categories_parts = ["Hand"]
# categories_fracture = ['fractured', 'normal']

# # Get image and model name; the default model is "Parts"
# # Parts - bone type predict model of 3 classes
# # Otherwise - fracture predict for each part
# def predict(img, model="Parts"):
#     size = 224

#     # Check if the provided model name is valid
#     if model not in ['Parts', 'Hand']:
#         raise ValueError("Invalid model name. Choose 'Parts' or 'Hand'.")

#     # Choose the appropriate model
#     chosen_model = model_parts if model == 'Parts' else model_hand_frac

#     # Load image with 224x224 pixels (the training model image size, RGB)
#     temp_img = image.load_img(img, target_size=(size, size))
#     x = image.img_to_array(temp_img)
#     x = np.expand_dims(x, axis=0)
#     images = np.vstack([x])
    
#     # Get the prediction
#     prediction = np.argmax(chosen_model.predict(images), axis=1)

#     # Choose the category and get the string prediction
#     prediction_str = categories_parts[prediction.item()] if model == 'Parts' else categories_fracture[prediction.item()]

#     return prediction_str



