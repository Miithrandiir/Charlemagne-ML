from typing import List

import numpy as np
import pandas
import torch

import lib
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sn


class NeuralNetworkClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(NeuralNetworkClassificationModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, 310)

        self.hidden_layer1 = nn.Linear(310, 310)
        self.hidden_layer2 = nn.Linear(310, 310)
        self.hidden_layer3 = nn.Linear(310, 310)
        self.hidden_layer4 = nn.Linear(310, 155)
        self.hidden_layer5 = nn.Linear(155, 155)
        self.hidden_layer6 = nn.Linear(155, 77)

        self.output_layer = nn.Linear(77, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.input_layer(x))
        # hiden : List[nn.Module] = self.hidden_layers;
        # for c in hiden:z
        #     out = self.relu(c(out))
        # out = self.relu(self.hidden_layers(out))
        out = self.relu(self.hidden_layer1(out))
        out = self.relu(self.hidden_layer2(out))
        out = self.relu(self.hidden_layer3(out))
        out = self.relu(self.hidden_layer4(out))
        out = self.relu(self.hidden_layer5(out))
        out = self.relu(self.hidden_layer6(out))
        out = self.output_layer(out)
        return out


dataframe = lib.load_data('data/result.csv')
(x_train, x_test, y_train, y_test), nbColumns = lib.extract_xy(dataframe)
x_train, x_test, y_train, y_test = lib.normalize(x_train, x_test, y_train, y_test)

model = NeuralNetworkClassificationModel(nbColumns, 17)

learning_rate = 0.01
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 2000
train_losses = np.zeros(num_epochs)
test_losses = np.zeros(num_epochs)

train_losses, test_losses = lib.train_network(model, optimizer, criterion, x_train, y_train, x_test, y_test, num_epochs,
                                              train_losses, test_losses)

plt.figure(figsize=(10, 10))
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

predictions_train = []
predictions_test = []
with torch.no_grad():
    predictions_train = model(x_train)
    predictions_test = model(x_test)

train_acc = lib.get_accuracy_multiclass(predictions_train, y_train)
test_acc = lib.get_accuracy_multiclass(predictions_test, y_test)

print(f"Training Accuracy: {round(train_acc * 100, 3)}")
print(f"Test Accuracy: {round(test_acc * 100, 3)}")

torch.save(model.state_dict(), "torch.pt")
torch.save(model, "torch_all.pt")

model_scripted = torch.jit.script(model)
model_scripted.save("traced_resnet_model.pt")