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

print("hi git")
print("hi git")

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
            image_path = os.path.join(self.path_dir_1, self.dir1_list[idx])
        else:
            class_id = 1
            idx -= len(self.dir1_list)
            image_path = os.path.join(self.path_dir_2, self.dir2_list[idx])

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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



# пишу модель
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # пишем свсвертки
        self.activ = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv0 = nn.Conv2d(3, 32, 3, stride=1, padding=0)
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)

        self.adaptivepool = nn.AdaptiveAvgPool2d((1, 1)) # говорю что в итоге хочу иметь размер 1 х 1
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 10)
        self.linear2 = nn.Linear(10, 2)  # 2 потому что в конце должно быть два класса (кошка или собака)

    # прогоняем свертки
    def forward(self, x):
        out = self.conv0(x)
        out = self.activ(out)  # после каждого слоя нужно писать функцию активации, чтобы вносить нелинейность
        out = self.maxpool(out)

        out = self.conv1(out)
        out = self.activ(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.activ(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.activ(out)

        out = self.adaptivepool(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.activ(out)
        out = self.linear2(out)

        return out



# фукция точности
def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1) # .detach() - отрубаем граф
    return answer.mean()



# функция подсчета параметром нейросети
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


train_dogs_path = '/Users/aroslavsapoval/Desktop/kagglecatsanddogs/PetImages/Dog'
train_cats_path = '/Users/aroslavsapoval/Desktop/kagglecatsanddogs/PetImages/Cat'

test_dogs_path = ''
test_cats_path = ''

train_ds_catdogs = Dataset2class(train_dogs_path, train_cats_path)
# test_ds_catdogs = Dataset2class(test_dogs_path, test_cats_path)


# пишу даталоадер
batch_size = 16
train_dataloader = torch.utils.data.DataLoader(
    train_ds_catdogs, batch_size=batch_size
    # shuffle=True, num_workers=1, drop_last=True - не работает что-то из этого
    # drop_last=True - выбрасываем последний элемент
)
# test_dataloader = torch.utils.data.DataLoader(
#     test_ds_catdogs, batch_size=batch_size
#     # shuffle=True, num_workers=1, drop_last=False - не работает что-то из этого
#     # drop_last=True - выбрасываем последний элемент
# )




model = ConvNet()
print("Количество параметров в нейронной сети", count_parameters(model))

# for sample in train_dataloader:
#     img = sample['img']
#     label = sample['label']
#     model(img)
#     break

# Оптимизатор, лосс функция, метрика
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))



# 5 этап --> функция обучения
i = 0
epochs = 10
for epoch in range(epochs):
    loss_val = 0
    acc_val = 0
    # (pbar := tqdm(
    for sample in train_dataloader:
        i+=1
        image, label = sample['img'], sample['label']
        optimizer.zero_grad()

        label = F.one_hot(label, 2).float() # 2 выходных слоя
        pred = model(image)

        loss = loss_fn(pred, label)
        loss.backward()

        loss_item = loss.item()
        loss_val += loss_item

        optimizer.step()
        acc_current = accuracy(pred, label)
        acc_val += acc_current
        # pbar.set_description(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')  # в реальном времени показываем
        print(f'итерация - {i} loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')  # в реальном времени показываем
        # ошибку и точность
    print(loss_val / len(train_dataloader))
    print(acc_val / len(train_dataloader))



