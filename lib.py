import math

import numpy as np
import pandas
import tqdm
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file: str = 'data/result.csv') -> pandas.DataFrame:
    dataframe = pandas.read_csv('data/result.csv', sep=";")
    del dataframe["id"]
    del dataframe["level_white"]
    del dataframe["current_player"]
    del dataframe["white_pieces_k"]
    del dataframe["white_pieces_lost_k"]
    del dataframe["black_pieces_k"]
    del dataframe["black_pieces_lost_k"]
    return dataframe


def extract_xy(dataframe: pandas.DataFrame):
    dataframe.sample(frac=1)
    x = dataframe
    y = dataframe["level_diff"]

    del x["level_diff"]

    y = y + 8

    return train_test_split(x.values, y.values, test_size=0.8), len(x.columns)


def normalize(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # y_test = y_test + 8
    # y_train = y_train + 8

    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    return x_train, x_test, y_train, y_test


def train_network(model, optimizer, criterion, X_train, y_train, X_test, y_test, num_epochs, train_losses, test_losses):
    for epoch in tqdm.trange(num_epochs):
        # clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()

        # forward feed
        y_pred = model(X_train)

        # calculate the loss
        loss_train = criterion(y_pred, y_train)

        # backward propagation: calculate gradients
        loss_train.backward()

        # update the weights
        optimizer.step()

        output_test = model(X_test)
        loss_test = criterion(output_test, y_test)

        train_losses[epoch] = loss_train.item()
        test_losses[epoch] = loss_test.item()



        if (epoch + 1) % 25 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}")

    return train_losses, test_losses

def get_accuracy_multiclass(pred_arr,original_arr):
    if len(pred_arr)!=len(original_arr):
        return False
    pred_arr = pred_arr.numpy()
    original_arr = original_arr.numpy()
    final_pred= []
    # we will get something like this in the pred_arr [32.1680,12.9350,-58.4877]
    # so will be taking the index of that argument which has the highest value here 32.1680 which corresponds to 0th index
    for i in range(len(pred_arr)):
        final_pred.append(np.argmax(pred_arr[i]))
    final_pred = np.array(final_pred)
    count = 0
    #here we are doing a simple comparison between the predicted_arr and the original_arr to get the final accuracy
    for i in range(len(original_arr)):
        if final_pred[i] == original_arr[i]:
            count+=1
    return count/len(final_pred)