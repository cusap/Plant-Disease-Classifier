import os
import random
from PIL import Image

dis_vec = []


def get_classes():
    classes_file = open("classes.txt", "a")
    for disease in os.listdir("./dataset"):
        dis_vec.append(disease)
        classes_file.write(disease)
        classes_file.write('\n')
    print(dis_vec)
    classes_file.close()


def make_yolo_txt():
    os.mkdir("./dataset-all")
    os.mkdir("./dataset-yolo")
    i = 0
    for class_name in dis_vec:
        os.mkdir("./dataset-yolo/{}".format(class_name))
        for image in os.listdir("./dataset/{}".format(class_name)):
            if image.endswith(".JPG"):
                name = "{}.txt".format(image[:-4])
                # random_left = round(random.uniform(0.5, 0.999999), 6)
                # random_right = round(random.uniform(0.5, 0.999999), 6)
                # input_string = "{} 0.500000 0.500000 {} {}".format(i, random_left, random_right)
                input_string = "{} 0.500000 0.500000 0.999999 0.999999".format(i)
                yolo_file = open("./dataset-yolo/{}/{}".format(class_name, name), "a")
                yolo_file.write(input_string)
                yolo_file.write('\n')
                yolo_file_all = open("./dataset-all/{}".format(name), "a")
                yolo_file_all.write(input_string)
                yolo_file_all.write('\n')
                im = Image.open("./dataset/{}/{}".format(class_name, image))
                rgb_im = im.convert('RGB')
                new_size = im.size
                rgb_im = rgb_im.resize(new_size)
                rgb_im.save("./dataset-all/{}.jpg".format(image[:-4]))
        print(class_name, i)
        i+=1


def count_images():
    i = 0
    for class_name in dis_vec:
        j = 0
        for image in os.listdir("./dataset/{}".format(class_name)):

            if image.endswith(".JPG"):
                j+=1
        print(class_name, j)
        i+=j
    print(i)

if __name__ == '__main__':
    get_classes()
    make_yolo_txt()
    # count_images()