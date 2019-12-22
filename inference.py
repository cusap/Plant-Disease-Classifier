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



proj_dir = os.path.dirname(__file__)
im_path = os.path.join(proj_dir, "my_im.jpg")
print(run_inf(im_path))