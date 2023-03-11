# #1
# import matplotlib.pyplot as plt  # построение графиков
# import torch
# X = torch.rand (1200,2)
# Y = (torch.sum((X - 0.5)**2, axis=1) < 0.1).float().view(-1,1)
# plt.figure(figsize=(5, 5))  # размеры (квадрат)
# plt.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=Y.numpy()[:, 0], s=30, cmap=plt.cm.Paired, edgecolors='k')
# plt.show()  # выводим рисунок

#2
from torch import nn
class Network(nn.Module):
    def __init__(self):
        super().__init__()

    # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
    # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)

    # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


def forward(self, x):
    # Pass the input tensor through each of our operations
    x = self.hidden(x)
    x = self.sigmoid(x)
    x = self.output(x)
    x = self.softmax(x)

    return x

# Create the network and look at it's text representation
model = Network()
print(model)

#3
# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)