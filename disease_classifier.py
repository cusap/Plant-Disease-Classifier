import os
import numpy as np
from PIL import Image
import glob

segmented_path = r"C:\Users\minht\PycharmProjects\Deep Learning\final_proj\PlantVillage-Dataset\raw\segmented\Apple*"

def get_data():
    image_list = []
    label_list =[]
    for label_name in glob.glob(segmented_path):
        label = label_name.split('\\')[-1]
        print(label)
        for pic_name in glob.glob(label_name + "\*"):
            im = Image.open(pic_name)
            im.load()
            image_list.append(np.asarray(im, dtype='int32'))
            label_list.append(label)
    return image_list, label_list

if __name__ == '__main__':
    train_im, train_labels = get_data()

    print(len(train_im))
    print(train_im[0].shape)
    print(train_labels[0])