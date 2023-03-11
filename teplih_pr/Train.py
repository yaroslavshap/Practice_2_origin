import torch
import numpy as np
import torch.nn as nn
from teplih_pr.Model import MyModel


def gen(shape, batch=1):
    imgs = []
    labels = []
    while True:
        for _ in range(batch):
            img = np.random.rand(*shape)
            label = np.random.choice([0, 1], size=10)
            imgs.append(img)
            labels.append(label)

        yield np.stack(imgs, axis=0), np.stack(labels, axis=0)
        imgs = []
        labels = []


model = MyModel()
device = torch.device("mps")
model.to(device)

opti = torch.optim.Adam(params=model.parameters(), lr=10e-3)
loss_fun = nn.MSELoss()

shape = (512, 512, 3)
# mps_device = torch.device("mps")
# model.to(mps_device)
# model.cpu()
# model.cuda(torch.device("mps"))
model.train()
count = 500

for i, batch in enumerate(gen(shape, batch=5)):
    image, label = batch
    img_tensor = torch.from_numpy(image.transpose(0, 3, 1, 2)).float().to(device)
    label_tensor = torch.from_numpy(label).float().to(device)
    # img_tensor = torch.from_numpy(image.transpose(0, 3, 1, 2)).float()
    # label_tensor = torch.from_numpy(label).float()
    opti.zero_grad()

    res = model(img_tensor)
    loss = loss_fun(res, label_tensor)
    loss.backward()
    opti.step()
    print(f'iter = {i}, loss - {loss.detach().numpy()}')
    if i == count:
        break
