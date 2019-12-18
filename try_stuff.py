from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
#tf.enable_eager_execution()

import tensorflow_hub as hub
import os
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

winnie = 1

if winnie:
    path_to_parent = r"/home/winnie/dhvanil/cgml/plant-classifier"
    segmented_path = path_to_parent + r"/PlantVillage-Dataset/raw/segmented"
    cp_path = path_to_parent + r"/Plant-Disease-Classifier/the_end/{epoch:04d}.cpkt"
    #cp_path = path_to_parent + r"/Plant-Disease-Classifier/model-checkpoints/{epoch:04d}.cpkt"
else:
    path_to_parent = r"C:\Users\minht\PycharmProjects\Deep Learning\final_proj"
    segmented_path = path_to_parent + r"\PlantVillage-Dataset\raw\segmented"
    cp_path = path_to_parent + r"\Plant-Disease-Classifier\model-checkpoints\{epoch:04d}.cpkt"


zip_file=tf.keras.utils.get_file(origin='https://storage.googleapis.com/plantdata/PlantVillage.zip',
 fname='PlantVillage.zip', extract=True)

data_dir = os.path.join(os.path.dirname(zip_file), 'PlantVillage')
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')




import time
from os.path import exists

def count(dir, counter=0):
    "returns number of files in dir and subdirs"
    for pack in os.walk(dir):
        for f in pack[2]:
            counter += 1
    return dir + " : " + str(counter) + "files"

import json

with open('Plant-Diseases-Detector-master/categories.json', 'r') as f:
    cat_to_name = json.load(f)
    classes = list(cat_to_name.values())

IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 64 #@param {type:"integer"}
# Inputs are suitably resized for the selected module. Dataset augmentation (i.e., random distortions of an image each time it is read) improves training, esp. when fine-tuning.

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    shuffle=False,
    color_mode="rgb",
    class_mode="categorical",
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE)

do_data_augmentation = True  # @param {type:"boolean"}
if do_data_augmentation:
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=360,
        horizontal_flip=True,
        vertical_flip = True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range = [.2, 1.0],
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')
else:
    train_datagen = validation_datagen

train_generator = train_datagen.flow_from_directory(
    train_dir,
    subset="training",
    shuffle=True,
    color_mode="rgb",
    class_mode="categorical",
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE)

model = tf.keras.Sequential([
  hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
                 output_shape=[1280],
                 trainable=False),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(rate=0.2),
  tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

#Compile model specifying the optimizer learning rate

LEARNING_RATE = 0.001 #@param {type:"number"}

cp_dir = os.path.dirname(cp_path)
cp_callback = ModelCheckpoint(filepath=cp_path, save_weights_only=True, save_best_only=True, period=2,
                                  verbose=1)

model.compile(
   optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
   loss='categorical_crossentropy',
   metrics=['accuracy'])


EPOCHS=10 #@param {type:"integer"}

history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples//train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples//validation_generator.batch_size,
        callbacks=[cp_callback],
        verbose= 1)
model.summary()
