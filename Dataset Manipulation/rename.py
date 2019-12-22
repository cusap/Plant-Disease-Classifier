import os
from PIL import Image

def rename():
    i = 0

    for filename in os.listdir("yolo-train-set"):
        im = Image.open("./yolo-train-set/"+filename)
        rgb_im = im.convert('RGB')
        rgb_im.save("./yolo-train/{}.jpg".format(i))
        i += 1


def resize():
    i = 0
    for filename in os.listdir("yolo-train-set"):
        im = Image.open("./yolo-train-set/"+filename)
        rgb_im = im.convert('RGB')
        new_size = im.size
        rgb_im = rgb_im.resize(new_size)
        rgb_im.save("./yolo-train-resized/{}.jpg".format(i))
        i += 1

if __name__ == '__main__':
    # Calling main() function
    # rename()
    resize()