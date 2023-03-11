import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2


# делаю кастомный датасет
class Dataset2class(torch.utils.data.Dataset):
    def __init__(self, path_dir_1: str, path_dir_2: str):
        super(Dataset2class, self).__init__()
        # запоминаю путь к папкам
        self.path_dir_1 = path_dir_1
        self.path_dir_2 = path_dir_2

        # плучаю список изображениий
        self.dir1_list = sorted(os.listdir(path_dir_1))
        self.dir2_list = sorted(os.listdir(path_dir_2))

    # получаю индекс
    def __getitem__(self, idx):
        if idx < len(self.dir1_list):
            class_id = 0
            self.image_path = os.path.join(self.path_dir_1, self.dir1_list[idx])
        else:
            class_id = 1
            idx -= len(self.dir1_list)
            self.image_path = os.path.join(self.path_dir_2, self.dir2_list[idx])
            print(self.image_path)

        img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.title({idx})
        # plt.imshow(img)
        # plt.show()
        img = img.astype(np.float32)
        img = img / 255.0
        img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
        # делаем из изображения тензоры
        img = img.transpose(2, 0, 1)
        t_image = torch.from_numpy(img)
        t_class_id = torch.tensor(class_id)
        return {'img': t_image, 'label': t_class_id}

    # узнаю длинну датасета
    def __len__(self):
        return len(self.dir1_list) + len(self.path_dir_2)





path = '/Users/aroslavsapoval/Desktop/kagglecatsanddogs/PetImages/Dog/.DS_Store'
if os.path.exists(path):
    print(path, 'существует')
    os.remove(path)
else:
    print("не существует")




train_dogs_path = '/Users/aroslavsapoval/Desktop/kagglecatsanddogs/PetImages/Dog'
train_cats_path = '/Users/aroslavsapoval/Desktop/kagglecatsanddogs/PetImages/Cat'
ds = Dataset2class(train_dogs_path, train_cats_path)




i = 0
while i<15000:
    try:
        sample = ds.__getitem__(i)
        if os.path.exists(ds.image_path):
            print(path, 'существует')
        else:
            print("не существует")
        # print("img - ", sample['img']," \n", "sample - ", sample['label'])
        print(i, " - ", ds.image_path)
    except:
        cv2.error
        os.remove(ds.image_path)
        FileNotFoundError
        # shutil.move(ds.image_path, '/Users/aroslavsapoval/Desktop/photo_not_work')
        continue
    i+=1