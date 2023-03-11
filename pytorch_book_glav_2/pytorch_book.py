import os
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

# работаю на GPU
device = torch.device("mps")

# функция отображения изображеня
def image_display(path_f):
    a = str(labels[prediction])
    im = cv2.imread(path_f)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.title(f' предсказание - {a}')
    plt.imshow(im)
    plt.show()


# функция соединения путей до изображений нужной категории в один список
def connect_paths_into_one_list(path3, path4):
    list3 = os.listdir(path3)
    list4 = os.listdir(path4)
    list_comm = []
    for i in list3:
        image_com = os.path.join(path3, str(i))
        list_comm.append(image_com)
    for i in list4:
        image_com = os.path.join(path4, str(i))
        list_comm.append(image_com)
    random.shuffle(list_comm)  # перемешал изображения в списке
    return list_comm


# проверяю существует ли файл
def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False


# функция для обучения модели
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device=device):
    for epoch in range(1, epochs + 1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in (tqdm(train_loader)):
            optimizer.zero_grad()  # обнуляем градиенты
            inputs, targets = batch  # берем пакет из каждого набора данных
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)  # пропускаем его через модель
            loss = loss_fn(output, targets)  # вычисляем потери от ожидаемого результата
            loss.backward()  # вычислляем градиенты
            optimizer.step()  # используем вычисленные градиенты для перехода на след шаг
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)
        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(
            epoch, training_loss, valid_loss, num_correct / num_examples))


# проверяю можно ли работать на GPU
def gpu_available():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(device)
    else:
        print("MPS device not found.")


# создание обучающего набора данных
train_data_path = "/Users/aroslavsapoval/Jupiter_Projects/book_project_1/train/"
val_data_path = "/Users/aroslavsapoval/Jupiter_Projects/book_project_1/val/"
test_data_path = "/Users/aroslavsapoval/Jupiter_Projects/book_project_1/test/"

img_transforms = transforms.Compose([transforms.Resize((64, 64)),  # меняем размер изображений
                                     transforms.ToTensor(),  # делаем из изображения тензор
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# делаем данные пригодные для обучения
train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=img_transforms, is_valid_file=check_image)
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=img_transforms, is_valid_file=check_image)
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=img_transforms, is_valid_file=check_image)



# пишем загрузчики данных
batch_size = 64  # сколько изображений пройдут через сеть прежде чем мы обучим ее и обновим
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


# класс модели
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(12288, 84) # входные данные
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 2)

    # какие данные передаются по сети при обучении и предсказании
    def forward(self, x):
        # x = self.flatten(x)
        x = x.view(-1, 12288)  # трехмерный тензор преобразовываем в одномерный
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # функция потерь определяет верный пргоноз или не верный
        return x

# функция, чтобы посчитать количество параметров нейронной сети
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

simplenet = SimpleNet()
print(count_parameters(simplenet)) # количество параметров в нейронной сети
optimizer = optim.Adam(simplenet.parameters(), lr=0.001) # оптимизатор
simplenet.to(device)


# обучаем модель
train(simplenet, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader, epochs=5, device=device)
labels = ['cat', 'fish']

# выбираем рандомно итоговый файл
path3 = "/Users/aroslavsapoval/Jupiter_Projects/book_project_1/val/cat/"
path4 = "/Users/aroslavsapoval/Jupiter_Projects/book_project_1/val/fish/"
path_finish = random.choice(connect_paths_into_one_list(path3, path4))
print(path_finish)



# делаем предсказание
img = Image.open(path_finish)
img = img_transforms(img).to(device)
img = torch.unsqueeze(img, 0)

simplenet.eval()
prediction = F.softmax(simplenet(img), dim=1)
prediction = prediction.argmax()
print(labels[prediction])

image_display(path_finish) # отображаем получившийся результат




# def predict(net, x, y):
#     y_pred = net.forward(x)
#     plt.plot(x.numpy(), y.numpy(), 'o', c='g', label='То что должно быть')
#     plt.plot(x.numpy(), y_pred.data.numpy(), 'o', c='r', label='Предсказание сети')
#     plt.legend(loc='upper left')

