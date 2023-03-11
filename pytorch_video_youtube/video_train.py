import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import torchvision as tv
import torch.utils.data
from tqdm import tqdm
import torch.nn.functional as F

# 1 этап --> Данные (Датасет) + Трансформация, аугментация (если нужно)
img_transforms = tv.transforms.Compose([tv.transforms.ToTensor()])

ds_mnist = tv.datasets.MNIST('/Users/aroslavsapoval/PycharmProjects/praktika_2/pytorch_video_youtube/datasets',
                             transform=img_transforms)
print(ds_mnist[0][0].numpy()[0].shape)
plt.imshow(ds_mnist[0][0].numpy()[0])
plt.show()

# 2 этап --> Даталоадер
batch_size = 16
dataloader = torch.utils.data.DataLoader(
    ds_mnist,
    batch_size=batch_size)
# shuffle=True,
# num_workers=1,
# drop_last=True)  # shuffle=True - перемешиваем изображения

for image, label in dataloader:
    print(image.shape)
    print(label.shape)
    break


# 3 этап --> пишем модель
class Neural(nn.Module):
    def __init__(self):
        super(Neural, self).__init__()
        self.flat = nn.Flatten()  # разворачиваем изображение ???
        self.linear1 = nn.Linear(28 * 28, 100)
        self.linear2 = nn.Linear(100, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.flat(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 4 этап --> Оптимизатор, лосс функция, метрика
model = Neural()  # экземпляр модели
print(count_parameters(model))  # количество параметров в нейронной сети
loss_f = nn.CrossEntropyLoss()  # лосс функция
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,
                            momentum=0.9)  # оптимизатор, сущность, которая обновляет веса модели
def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1) # .detach() - отрубаем граф
    return answer.mean()


# 5 этап --> функция обучения
epochs = 10
for epoch in range(epochs):
    loss_val = 0
    acc_val = 0
    for image, label in (pbar := tqdm(dataloader)):
        optimizer.zero_grad()

        label = F.one_hot(label, 10).float()
        pred = model(image)

        loss = loss_f(pred, label)
        loss.backward()

        loss_item = loss.item()
        loss_val += loss_item
        
        optimizer.step()
        acc_current = accuracy(pred, label)
        acc_val+=acc_current
        pbar.set_description(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}') # в реальном времени показываем
        # ошибку и точность
    print(loss_val / len(dataloader))
    print(acc_val / len(dataloader))



