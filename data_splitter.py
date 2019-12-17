from shutil import copyfile
import glob
import random
path_to_parent = r"/home/winnie/dhvanil/cgml/plant-classifier"
segmented_path = path_to_parent + r"/PlantVillage-Dataset/raw/segmented"


def splitter():

    train_count =0
    val_count=0
    test_count=0
    for i,label_name in enumerate(glob.glob(segmented_path)):
        label = label_name.split('\\')[-1]
        print(label)
        for count, pic_name in enumerate(glob.glob(label_name + "\\*")):
            num = random.randint(1, 10)

            if num < 7:
                copyfile(pic_name, path_to_parent + "/train/" + label)
                train_count+=1
            elif num < 9:
                copyfile(pic_name, path_to_parent + "/val/" + label)
                val_count+=1
            else:
                copyfile(pic_name, path_to_parent + "/test/" + label)
                test_count+=1
    print("train images: " + str(train_count))
    print("val images: " + str(val_count))
    print("test images: " + str(test_count))

splitter()