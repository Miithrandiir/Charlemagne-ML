import numpy as np
import pandas
import seaborn as sn
import torch
import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file: str = 'data/result_stockfish.csv') -> pandas.DataFrame:
    dataframe = pandas.read_csv(file, sep=";")
    # dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    dataframe = dataframe[dataframe["current_player"] == "white"]
    dataframe = dataframe[dataframe["current_step"] >= 100]
    dataframe["level_diff"] = dataframe["level_white"]
    del dataframe["id"]
    del dataframe["level_white"]
    del dataframe["current_player"]
    del dataframe["level_back"]
    del dataframe["current_step"]
    # del dataframe["cp_diff"]
    # del dataframe["cp"]

    # del dataframe["white_pieces_k"]
    # del dataframe["black_pieces_k"]

    pieces = ["b", "n", "p", "q", "r", "k"]
    for i in pieces:
        del dataframe["white_pieces_lost_" + i]
        del dataframe["white_pieces_" + i]
        del dataframe["black_pieces_lost_" + i]
        del dataframe["black_pieces_" + i]

    # dataframe["white_pieces"] = 0
    # dataframe["black_pieces"] = 0
    # # dataframe["black_pieces_lost"] = 0
    # # dataframe["white_pieces_lost"] = 0
    # for i in pieces:
    #     dataframe["white_pieces"] += dataframe["white_pieces_" + i]
    #     dataframe["black_pieces"] += dataframe["black_pieces_" + i]
    #     # dataframe["white_pieces_lost"] += dataframe["white_pieces_lost_" + i]
    #     # dataframe["black_pieces_lost"] += dataframe["black_pieces_lost_" + i]
    #
    # new_dataframe = pandas.DataFrame()
    # dataframe = new_dataframe.assign(level_black=dataframe["level_back"].values,
    #                                  cp_diff=dataframe["cp_diff"].values,
    #                                  cp=dataframe["cp"].values,
    #                                  white_pieces=dataframe["white_pieces"].values,
    #                                  black_pieces=dataframe["black_pieces"].values,
    #                                  # white_pieces_lost=dataframe["white_pieces_lost"].values,
    #                                  # black_pieces_lost=dataframe["black_pieces_lost"].values,
    #                                  current_step=dataframe["current_step"].values,
    #                                  level_diff=dataframe["level_diff"].values)
    return dataframe


def extract_xy(dataframe: pandas.DataFrame):
    # dataframe.sample(frac=1)
    x = dataframe
    # x["current_player"] = x["current_player"].map(lambda z : 0 if z == "white" else 1)

    y = dataframe["level_diff"]

    del x["level_diff"]

    y = y - 1
    # y.loc[y < 0] = 2
    # y.loc[y > 0] = 1

    return train_test_split(x.values, y.values, test_size=0.5), len(x.columns)


def normalize(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()

    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)
    # y_test = y_test + 8
    # y_train = y_train + 8

    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    return x_train, x_test, y_train, y_test


def correlation_matrix(dataframe: pandas.DataFrame):
    fig, ax = plt.subplots(1, 1, figsize=(15, 15), dpi=300)

    sn.heatmap(dataframe.corr(), cmap="PiYG", vmin=-0.5, vmax=0.5, mask=[False for i in range(len(dataframe.columns))],
               linewidths=.5)

    ax.set_ylabel('')
    ax.set_xlabel('')

    plt.tight_layout()
    plt.show()


def train_network(model, optimizer, criterion, X_train, y_train, X_test, y_test, num_epochs, train_losses, test_losses):
    train_mse = np.zeros(num_epochs)
    test_mse = np.zeros(num_epochs)
    # mse = nn.MSELoss()
    for epoch in tqdm.trange(num_epochs):
        optimizer.zero_grad()

        # forward feed
        y_pred = model(X_train)
        y_pred.cuda()
        # calculate the loss
        loss_train = criterion(y_pred, y_train)
        # clear out the gradients from the last step loss.backward()
        # backward propagation: calculate gradients
        loss_train.backward()
        # update the weights
        optimizer.step()
        with torch.no_grad():
            model.eval()
            output_test = model(X_test)
            loss_test = criterion(output_test, y_test)

            train_losses[epoch] = loss_train.item()
            test_losses[epoch] = loss_test.item()

        # mse_test = mse(torch.max(y_pred, 1).values, y_train)
        #
        # train_mse[epoch] = mse_test.item()
        # test_mse[epoch] = mse(torch.max(output_test, 1).values, y_test).item()

        if (epoch + 1) % 25 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}")

    return train_losses, test_losses, test_mse, train_mse

def get_accuracy_multiclass(pred_arr, original_arr):
    if len(pred_arr) != len(original_arr):
        return False
    pred_arr = pred_arr.numpy()
    original_arr = original_arr.numpy()
    final_pred = []
    # we will get something like this in the pred_arr [32.1680,12.9350,-58.4877]
    # so will be taking the index of that argument which has the highest value here 32.1680 which corresponds to 0th index
    for i in range(len(pred_arr)):
        final_pred.append(np.argmax(pred_arr[i]))
    final_pred = np.array(final_pred)
    count = 0
    # here we are doing a simple comparison between the predicted_arr and the original_arr to get the final accuracy
    for i in range(len(original_arr)):
        if final_pred[i] == original_arr[i]:
            count += 1
    return count / len(final_pred)
