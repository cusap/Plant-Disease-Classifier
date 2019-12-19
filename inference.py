import tensorflow as tf
import tensorflow_hub as hub
import os
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import sklearn
from PIL import Image
import numpy as np

def run_inf(im_path):
    proj_dir = os.path.dirname(__file__)
    weight_dir = os.path.join(proj_dir, r"/class_weights")
    cp_path = proj_dir + r"/class_weights/{epoch:04d}.cpkt"
    learning_rate = 1e-4
    im_shape = (224,224,3)

    model = tf.keras.Sequential([
      hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
                     output_shape=[1280],
                     trainable=False),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.Dense(38, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                              optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                              metrics=['accuracy'])

    cp_dir = os.path.dirname(weight_dir)
    cp_callback = ModelCheckpoint(filepath=cp_path, save_weights_only=True, save_best_only=True, period=2,
                                      verbose=1)

    model.load_weights(proj_dir + r"/class_weights/0006.cpkt")
    inf_image = Image.open(im_path).resize(im_shape)
    inf_image = np.array(inf_image)/255.0
    result = model.predict(inf_image[np.newaxis,...])
    predicted_class = np.argmax(result[0], axis=-1)
    classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
               'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
               'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
               'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
               'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    return classes[predicted_class], np.amax(result)

proj_dir = os.path.dirname(__file__)
im_path = os.path.join(proj_dir, "my_im.jpg")
print(run_inf(im_path))