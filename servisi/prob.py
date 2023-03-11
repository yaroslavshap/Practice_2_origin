import os
import shutil
import random
import cv2
from matplotlib import pyplot as plt


# создаем новую папку
def creating_a_folder(name):
    try:
        papka = os.mkdir(name)
        print(papka)
    except:
        FileExistsError


# проверяю существует ли папка
def folder_exists(destination1, name):
    path1 = os.path.join(destination1, name)
    if os.path.isdir(path1):
        print("это папка и она существует")


# переносим папку в другое место
def moving_folder(destination2, name):
    try:
        path = shutil.move(name, destination2)
        print(path)
    except:
        shutil.Error


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


# перемешиваю все элементы в папке и вывожу их
def shuffling_folder(common):
    list_com = os.listdir(common)
    random.shuffle(list_com)  # перемешал изображения в списке

    for i in range(len(list_com)):
        try:
            print("изображение - ", i, " - ", list_com[i])
            image = os.path.join(common, str(list_com[i]))
            print(image, "\n")
            # im = cv2.imread(image)
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # plt.imshow(im)
            # plt.show()
        except:
            cv2.error
            continue


# удаляю папку
def delete_folder(common):
    shutil.rmtree(common)
    print("Папка - ", common, " - удалена")



name = 'common_cat_fish_train' # имя новой папки
destination1 = '/Users/aroslavsapoval/PycharmProjects/praktika_2/' # куда сохранилась папка
destination2 = '/Users/aroslavsapoval/Jupiter_Projects/book_project_1/' # куда переношу папку
creating_a_folder(name)
folder_exists(destination1, name)
moving_folder(destination2, name)

path1 = "/Users/aroslavsapoval/Downloads/cats_dogs_light"
common = '/Users/aroslavsapoval/Jupiter_Projects/book_project_1/train/cat'
common1 = copy_to_folder(path1, common)
shuffling_folder(common)
a = input("удалить папку? [1 - да / любая клавиша - нет]")
if a == '1':
    delete_folder(common)

