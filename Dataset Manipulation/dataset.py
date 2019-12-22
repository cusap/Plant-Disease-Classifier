import os
import shutil

def partial_write_corrector():
    filepath = "./flies.txt"
    dataset_path = "./dataset-all"
    completed_path = "./dataset-all-sent"
    # os.mkdir(completed_path)
    print(filepath)
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 0
        while line:
            current_file = "{}/{}".format(dataset_path, line.strip())
            new_path = "{}/{}".format(completed_path, line.strip())
            print("Moving {} to {}".format(current_file, new_path))
            shutil.move(current_file, new_path)
            # print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
            cnt += 1
        print("Moved {} Files".format(cnt))


if __name__ == '__main__':
    partial_write_corrector()