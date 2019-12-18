import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import numpy as np
import tensorflow as tf
import seaborn as sn
from PIL import Image
import pandas as pd
import random
import glob
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, GlobalAveragePooling2D, \
    AveragePooling2D, BatchNormalization, Add, \
    DepthwiseConv2D, ReLU, Reshape, MaxPooling2D
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import tensorflow_hub as hub


PERCENT_TRAIN = .8
lamb = .000001
winnie = 1

if winnie:
    path_to_parent = r"/home/winnie/dhvanil/cgml/plant-classifier"
    segmented_path = path_to_parent + r"/PlantVillage-Dataset/raw/segmented"
    cp_path = path_to_parent + r"/Plant-Disease-Classifier/to_comp/{epoch:04d}.cpkt"
    cp_dir = path_to_parent + r"/Plant-Disease-Classifier/the_end"
    #cp_path = path_to_parent + r"/Plant-Disease-Classifier/model-checkpoints/{epoch:04d}.cpkt"
else:
    path_to_parent = r"C:\Users\minht\PycharmProjects\Deep Learning\final_proj"
    segmented_path = path_to_parent + r"\PlantVillage-Dataset\raw\segmented"
    cp_path = path_to_parent + r"\Plant-Disease-Classifier\model-checkpoints\{epoch:04d}.cpkt"


# learning_rate = .045
learning_rate = .001
lr_decay = .98
batch_size = 32
epochs = 200
sample_ratio = 16
data_shape = (224, 224, 3)
num_cat = 38


train_dir = path_to_parent + r"/train"
val_dir = path_to_parent + r"/val"
test_dir = path_to_parent + r"/test"

def shuffle(data_im, data_labels):
    shuffled_index = np.arange(len(data_labels))
    np.random.shuffle(shuffled_index)
    return data_im[shuffled_index], data_labels[shuffled_index]


def format_data(train_images, train_labels, num_category):
    train_images = tf.image.resize(train_images, (224, 224))
    print(train_images.shape)

    train_labels = train_labels.astype('float32')
    train_labels = tf.keras.utils.to_categorical(train_labels, num_category)
    return train_images, train_labels


def open_data():
    image_list = []
    label_list = []
    label_names = []
    for i, label_name in enumerate(glob.glob(segmented_path)):
        label = label_name.split('/')[-1]
        print(label)
        label_names.append(label)
        for count, pic_name in enumerate(glob.glob(label_name + "/*")):
            if random.randint(1, sample_ratio) == 1:
                try:
                    im = Image.open(pic_name)
                    im.load()
                    new_im = np.asarray(im, dtype='float32')
                    new_im = new_im / 255
                    image_list.append(new_im)
                    label_list.append(i)

                except:
                    print("bad image")
                    continue
    image_list, label_list = shuffle(image_list, label_list)
    return np.asarray(image_list), np.asarray(label_list), label_names


def get_data():
    image_list, label_list, label_names = open_data()
    print('starting format')
    image_list, label_list = format_data(image_list, label_list, len(label_names))
    image_list, label_list = shuffle(image_list, label_list)
    border = int(len(image_list) * PERCENT_TRAIN)
    print(len(image_list))
    return image_list[0:border], label_list[0:border], image_list[border + 1:], label_list[border + 1:], label_names


def bottleneck_s1(orig, num_channels, exp_fac, resid=True):
    x = Conv2D(orig.get_shape()[-1] * exp_fac, kernel_size=(1, 1), strides=1, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(lamb),
               activity_regularizer=tf.keras.regularizers.l2(lamb))(orig)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    x = DepthwiseConv2D(3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    x = Conv2D(num_channels, kernel_size=(1, 1), strides=1, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(lamb),
               activity_regularizer=tf.keras.regularizers.l2(lamb), activation='linear')(x)
    if (resid):
        return Add()([x, orig])
    else:
        return x


def bottleneck_s2(x, num_channels, exp_fac):
    x = Conv2D(x.get_shape()[-1] * exp_fac, kernel_size=(1, 1), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    x = DepthwiseConv2D(3, padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    x = Conv2D(num_channels, kernel_size=(1, 1), strides=1, padding='same', activation='linear')(x)
    return x


def b_block(x, num_channels, exp_fac, n, s):
    if (s == 1):
        x = bottleneck_s1(x, num_channels, exp_fac, resid=False)
    else:
        x = bottleneck_s2(x, num_channels, exp_fac)

    for i in range(1, n):
        x = bottleneck_s1(x, num_channels, exp_fac)
    return x


def scheduler(epoch):
    lr = learning_rate * (lr_decay ** epoch)
    return lr



import time
from os.path import exists


import json




if __name__ == '__main__':
    try:
        with tf.device('/device:GPU:0'):

            IMAGE_SHAPE = (224, 224)

            BATCH_SIZE = 64  # @param {type:"integer"}

            zip_file = tf.keras.utils.get_file(origin='https://storage.googleapis.com/plantdata/PlantVillage.zip',
                                               fname='PlantVillage.zip', extract=True)

            data_dir = os.path.join(os.path.dirname(zip_file), 'PlantVillage')
            train_dir = os.path.join(data_dir, 'train')
            validation_dir = os.path.join(data_dir, 'validation')





            with open('Plant-Diseases-Detector-master/categories.json', 'r') as f:
                cat_to_name = json.load(f)
                classes = list(cat_to_name.values())


            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1. / 255,
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                fill_mode='nearest')

            train_generator = train_datagen.flow_from_directory(
                train_dir,
                subset="training",
                shuffle=True,
                seed=42,
                color_mode="rgb",
                class_mode="categorical",
                target_size=IMAGE_SHAPE,
                batch_size=BATCH_SIZE)

            validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
            validation_generator = validation_datagen.flow_from_directory(
                validation_dir,
                shuffle=True,
                color_mode="rgb",
                class_mode="categorical",
                target_size=IMAGE_SHAPE,
                batch_size=BATCH_SIZE)

            # input = tf.keras.Input(shape=data_shape)
            # imported model code
            imported_model = tf.keras.applications.MobileNetV2(input_shape=data_shape, include_top=False,
                                                               weights='imagenet')
            imported_model.trainable = False
            # model = tf.keras.Sequential([imported_model,GlobalAveragePooling2D(), Dense(num_cat)])
            global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
            prediction_layer = tf.keras.layers.Dense(num_cat, activation='softmax')

            model = tf.keras.Sequential([
                hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
                               output_shape=[1280],
                               trainable=False),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
            ])

            # save model checkpoints

            cp_dir = os.path.dirname(cp_path)
            cp_callback = ModelCheckpoint(filepath=cp_path, save_weights_only=True, save_best_only=True, period=2,
                                          verbose=1)

            lr_scheduler = LearningRateScheduler(scheduler, verbose=1)

            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=.9, momentum=.9)

            model.compile(loss='categorical_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                          metrics=['accuracy'])

            recent = tf.train.latest_checkpoint(cp_dir)
            model.load_weights(path_to_parent + r"/Plant-Disease-Classifier/the_end/0008.cpkt")


            n = 30
            results = model.predict_generator(validation_generator, verbose=1)
            test_pred = np.array([np.argmax(x) for x in results])
            conf_mat = confusion_matrix(validation_generator.classes, test_pred>.5)

            labels = list(validation_generator.class_indices.keys())
            print(labels)

            df_cm = pd.DataFrame(conf_mat, index=labels,
                    columns=labels)
            plt.figure(figsize=(38,38))
            sn.heatmap(df_cm, annot=True)
            print(conf_mat)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()

    except RuntimeError as e:
        print(e)
