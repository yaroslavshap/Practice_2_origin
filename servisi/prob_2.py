import os
import shutil
import random
import cv2
from matplotlib import pyplot as plt

# копирую все в одну папку
def copy_to_folder(path1, common):
    list1 = os.listdir(path1)
    print(len(list1))

    for i in list1:
        try:
            image1 = os.path.join(path1, str(i))
            shutil.copy(image1, common)
        except:
            # IsADirectoryError
            continue
    print(len(os.listdir(common)))
    return common

path1 = "/Users/aroslavsapoval/Downloads/kagglecatsanddogs_5340/PetImages/Cat"
common = '/Users/aroslavsapoval/Jupiter_Projects/book_project_1/train/cat'
common1 = copy_to_folder(path1, common)