import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import lib


class NeuralNetworkClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(NeuralNetworkClassificationModel, self).__init__()
        max_layer = 200
        self.input_layer = nn.Linear(input_dim, max_layer)
        self.hidden_layer1 = nn.Linear(max_layer, max_layer)
        self.output_layer = nn.Linear(max_layer, output_dim)
        self.relu = nn.ReLU()

        self.input_layer = self.input_layer.cuda()
        self.hidden_layer1 = self.hidden_layer1.cuda()
        self.output_layer = self.output_layer.cuda()
        self.relu = self.relu.cuda()

    def forward(self, x):
        out = self.relu(self.input_layer(x))
        out = self.relu(self.hidden_layer1(out))
        out = self.output_layer(out)
        return out


dataframe = lib.load_data('data/result_stockfish.csv')
print(dataframe.columns)
lib.correlation_matrix(dataframe)
(x_train, x_test, y_train, y_test), nbColumns = lib.extract_xy(dataframe)
x_train, x_test, y_train, y_test = lib.normalize(x_train, x_test, y_train, y_test)

x_train = x_train.cuda()
y_train = y_train.cuda()
x_test = x_test.cuda()
y_test = y_test.cuda()

model = NeuralNetworkClassificationModel(nbColumns, 20)
model.parameters()
model.cuda()

learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
criterion.cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 1000
train_losses = np.zeros(num_epochs)
test_losses = np.zeros(num_epochs)

train_losses, test_losses, test_mse, train_mse = lib.train_network(model, optimizer, criterion, x_train, y_train,
                                                                   x_test, y_test, num_epochs,
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
    predictions_train = predictions_train.cpu()
    predictions_test = model(x_test)
    predictions_test = predictions_test.cpu()


x_train = x_train.cpu()
x_test = x_test.cpu()
y_train = y_train.cpu()
y_test = y_test.cpu()

train_acc = lib.get_accuracy_multiclass(predictions_train, y_train)
test_acc = lib.get_accuracy_multiclass(predictions_test, y_test)

print(f"Training Accuracy: {round(train_acc * 100, 3)}")
print(f"Test Accuracy: {round(test_acc * 100, 3)}")

torch.save(model.state_dict(), "torch.pt")
torch.save(model, "torch_all.pt")

model_scripted = torch.jit.script(model)
model_scripted.save("traced_resnet_model.pt")