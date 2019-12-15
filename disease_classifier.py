import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import random
import glob
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, GlobalAveragePooling2D,AveragePooling2D ,BatchNormalization, Add, \
    DepthwiseConv2D, ReLU, Reshape, MaxPooling2D
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PERCENT_TRAIN = .8
lamb = .000001
path_to_parent = r"/home/winnie/dhvanil/cgml/plant-classifier"
#path_to_parent = r"C:\Users\minht\PycharmProjects\Deep Learning\final_proj"
#segmented_path = path_to_parent + r"\PlantVillage-Dataset\raw\segmented"
segmented_path = path_to_parent + r"/PlantVillage-Dataset/raw/segmented"
#cp_path = path_to_parent + r"/Plant-Disease-Classifier/model-checkpoints/{epoch:04d}.cpkt"
#cp_path = path_to_parent + r"\Plant-Disease-Classifier\model-checkpoints\{epoch:04d}.cpkt"
cp_path = path_to_parent + r"/Plant-Disease-Classifier/conv-big/{epoch:04d}.cpkt"
#learning_rate = .045
learning_rate = .001
lr_decay = .98
batch_size = 32
epochs = 200
sample_ratio = 16
data_shape = (224,224,3)
num_cat = 38

def clean_dataset():

    for i,label_name in enumerate(glob.glob(segmented_path)):
        cnt = 0
        label = label_name.split('\\')[-1]
        print(label)
        for count, pic_name in enumerate(glob.glob(label_name + "\\*")):
            try:
                im = Image.open(pic_name)
                im.load()
                new_im = np.asarray(im, dtype='float32')
                new_im = new_im / 255
            except Exception as e:
                print(e)
                print("bleh")
                cnt += 1
        print("corrupted count: " +str(cnt))
def shuffle(data_im, data_labels):
    shuffled_index = np.arange(len(data_labels))
    np.random.shuffle(shuffled_index)
    return data_im[shuffled_index], data_labels[shuffled_index]

def format_data(train_images, train_labels, num_category):
    train_images = tf.image.resize(train_images, (224,224))
    print(train_images.shape)

    train_labels = train_labels.astype('float32')
    train_labels = tf.keras.utils.to_categorical(train_labels, num_category)
    return train_images, train_labels

def open_data():
    image_list = []
    label_list =[]
    label_names = []
    for i,label_name in enumerate(glob.glob(segmented_path)):
        label = label_name.split('/')[-1]
        print(label)
        label_names.append(label)
        for count, pic_name in enumerate(glob.glob(label_name + "/*")):
            if random.randint(1,sample_ratio)==1:
                try:
                    im = Image.open(pic_name)
                    im.load()
                    new_im = np.asarray(im, dtype='float32')
                    new_im = new_im/255
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
    border = int(len(image_list)*PERCENT_TRAIN)
    print(len(image_list))
    return image_list[0:border], label_list[0:border], image_list[border+1:], label_list[border+1:], label_names

def bottleneck_s1(orig, num_channels, exp_fac, resid=True):
    x = Conv2D(orig.get_shape()[-1]*exp_fac, kernel_size=(1,1), strides=1, padding='same',kernel_regularizer=tf.keras.regularizers.l2(lamb),
            activity_regularizer=tf.keras.regularizers.l2(lamb))(orig)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    x = DepthwiseConv2D(3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    x = Conv2D(num_channels, kernel_size=(1,1), strides=1, padding ='same',kernel_regularizer=tf.keras.regularizers.l2(lamb),
            activity_regularizer=tf.keras.regularizers.l2(lamb), activation='linear')(x)
    if(resid):
        return Add()([x,orig])
    else:
        return x

def bottleneck_s2(x, num_channels, exp_fac):

    x = Conv2D(x.get_shape()[-1]*exp_fac, kernel_size=(1, 1), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    x = DepthwiseConv2D(3, padding='same', strides= 2)(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    x = Conv2D(num_channels, kernel_size=(1, 1), strides= 1, padding='same', activation='linear')(x)
    return x

def b_block(x, num_channels, exp_fac, n, s):

    if(s==1):
        x = bottleneck_s1(x, num_channels, exp_fac, resid = False)
    else:
        x = bottleneck_s2(x, num_channels, exp_fac)

    for i in range(1, n):
        x = bottleneck_s1(x, num_channels, exp_fac)
    return x

def scheduler(epoch):
    lr = learning_rate * (lr_decay ** epoch)
    return lr


if __name__ == '__main__':
    clean_dataset()
    #train_im, train_labels, val_im, val_labels, label_names = get_data()
    #print(len(train_im))
    try:
        with tf.device('/device:GPU:0'):
            #input = tf.keras.Input(shape=data_shape)
            '''
            #basic
            x = Conv2D(32,3,3)(input)
            x = Conv2D(32,3,3)(x)
            x = Conv2D(32, 3, 3)(x)
            #x = Conv2D(32, 3, 2)(x)
            x = Flatten()(x)
            x = Dense(num_cat)(x)
            output = Activation('softmax')(x)
            model = tf.keras.Model(inputs=input, outputs=output)

            '''
            '''
            x = Conv2D(32, 3,strides= 2,padding='same')(input)

            x = b_block(x, num_channels = 16, exp_fac=1, n=1, s=1)

            x = b_block(x, num_channels = 24, exp_fac=6, n=2, s=2)

            x = b_block(x, num_channels = 32, exp_fac=6, n=3, s=2)
            x = b_block(x, num_channels = 64, exp_fac=6, n=4, s=2)
            x = b_block(x, num_channels = 96, exp_fac=6, n=3, s=1)

            x = b_block(x, num_channels=160, exp_fac=6, n=3, s=2)
            x = b_block(x, num_channels=320, exp_fac=6, n=1, s=1)
            x = Conv2D(1280, 1, strides=1, padding='same')(x)
            x = Dropout(.25)(x)
            x = AveragePooling2D(pool_size=(8,8))(x)
            x = Reshape((1,1,1280))(x)
            x = Conv2D(len(label_names),1, padding='same')(x)
            x = Dropout(.25)(x)

            x = Activation('softmax')(x)
            output = Reshape((len(label_names),))(x)
            #output = Reshape((len(label_names),))(x)
            
            model = tf.keras.Model(inputs=input, outputs=output)
            '''


            #imported model code
            imported_model = tf.keras.applications.MobileNetV2(input_shape=data_shape, include_top = False, weights='imagenet')
            '''
            x = imported_model.output
            x = GlobalAveragePooling2D()(x)
            print(x.get_shape)
            x = Dense(num_cat)(x)
            x = Dropout(.25)(x)
            output = Activation('softmax')(x)
        
            model = tf.keras.Model(inputs=input, outputs=output)
            '''

            imported_model.trainable = False
            '''
            for layer in imported_model.layers[:90]:
                layer.trainable = False
            '''
            #model = tf.keras.Sequential([imported_model,GlobalAveragePooling2D(), Dense(num_cat)])

            model = tf.keras.Sequential([imported_model,
                                         Flatten(),
                                         Dropout(.5),
                                         Dense(512,kernel_regularizer=tf.keras.regularizers.l2(lamb),
            activity_regularizer=tf.keras.regularizers.l2(lamb)),
                                         BatchNormalization(),
                                         ReLU(),
                                         Dropout(.5),
                                         Dense(num_cat, kernel_regularizer=tf.keras.regularizers.l2(lamb),
            activity_regularizer=tf.keras.regularizers.l2(lamb)),
                                         Activation('softmax')])

            '''
            #Some other implementation's code cuz im a dumb boy
            model = tf.keras.Sequential([
                hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
                               output_shape=[1280],
                               trainable=False),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(len(label_names), activation='softmax')
            ])
            '''
            #save model checkpoints

            cp_dir = os.path.dirname(cp_path)
            cp_callback = ModelCheckpoint(filepath=cp_path, save_weights_only=True, save_best_only=True, period=2,
                                              verbose=1)


            im_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=360,
                                                                     zoom_range=[0.5, 1.0],
                                                                     width_shift_range=0.2,
                                                                     height_shift_range=0.2,
                                                                     horizontal_flip=True,
                                                                     vertical_flip=True,
                                                                     data_format="channels_last",
                                                                     brightness_range = [.2, 1.0],
                                                                     shear_range = .2,
                                                                     fill_mode='nearest',
                                                                     rescale= 1/.255
                                                                     )

            '''
            im_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                                                                     zoom_range=[0.1, .1],
                                                                     width_shift_range=0.2,
                                                                     height_shift_range=0.2,
                                                                     horizontal_flip=True,
                                                                     vertical_flip=True,
                                                                     rescale=1./255,
                                                                     data_format="channels_last",
                                                                     brightness_range = [.2, 1.0],
                                                                     )
            '''
            val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
            no_aug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

            train_generator = im_gen.flow_from_directory(
                segmented_path,
                shuffle=True,
                color_mode="rgb",
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical')



            val_generator = val_gen.flow_from_directory(
                segmented_path,
                shuffle=True,
                color_mode="rgb",
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical')
            lr_scheduler = LearningRateScheduler(scheduler, verbose=1)

            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=.9, momentum=.9)

            model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
            '''
            model_log = model.fit(train_im, train_labels, batch_size=batch_size, epochs=epochs,
                                  callbacks=[lr_scheduler, cp_callback],
                                  validation_data=(val_im, val_labels), verbose=2)
                                  '''


            model.summary()

            model_log = model.fit_generator(train_generator, epochs=20,
                                            callbacks=[lr_scheduler, cp_callback],
                                            validation_data=val_generator, verbose=1)


    except RuntimeError as e:
        print(e)

