import cv2
import numpy as np
import os
import shutil
import wget
import tarfile
from itertools import compress
from matplotlib import pyplot as plt

#save data train and test funtion
def splitDatabase(folder2):

    categories = os.listdir(folder2)
    # Create target directory & all intermediate directories if don't exists
    dirName = os.path.join(folder2, "train")
    try:
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")

    for cat in categories:
        origin_path = os.path.join(folder2, cat)
        dest_dir = os.path.join(folder2, "train")
        if not origin_path == os.path.join(folder2, "BACKGROUND_Google"):
            shutil.move(origin_path, dest_dir)

    base_path = os.getcwd()
    data_path = os.path.join(base_path, "101_ObjectCategories/train")
    categories = os.listdir(data_path)
    test_path = os.path.join(base_path, "101_ObjectCategories/val")
    if not os.path.isdir(os.path.join(folder2, "val")):
        dirName = os.path.join(folder2, "val")
        try:
            os.makedirs(dirName)
            print("Directory " , dirName ,  " Created ")
        except FileExistsError:
            print("Directory " , dirName ,  " already exists")

    for cat in categories:
        image_files = os.listdir(os.path.join(data_path, cat))
        choices = np.random.choice([0, 1], size=(len(image_files),), p=[.70, .30])
        files_to_move = compress(image_files, choices)

        for _f in files_to_move:
            origin_path = os.path.join(data_path, cat,  _f)
            dest_dir = os.path.join(test_path, cat)
            dest_path = os.path.join(test_path, cat, _f)
            if not os.path.isdir(dest_dir):
                os.mkdir(dest_dir)

            shutil.move(origin_path, dest_path)
