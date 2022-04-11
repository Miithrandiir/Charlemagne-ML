import pandas
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import seaborn as sn

# Récupération des données

dataframe = pandas.read_csv('data/result.csv', sep=";")
del dataframe["id"]
del dataframe["level_white"]
del dataframe["current_player"]
del dataframe["white_pieces_k"]
del dataframe["white_pieces_lost_k"]
del dataframe["black_pieces_k"]
del dataframe["black_pieces_lost_k"]

# dataframe["current_player"] = 0 if dataframe["current_player"].all() == "white" else 1
#
# fig, ax = plt.subplots(1, 1, figsize=(15, 15), dpi=300)
# sn.heatmap(dataframe.corr(), cmap="PiYG", vmin=-0.5, vmax=0.5, mask=[False for i in range(len(dataframe.columns))],
#            linewidths=.5)
#
# ax.set_ylabel('')
# ax.set_xlabel('')
#
# plt.tight_layout()
# plt.show()

dataframe.sample(frac=1)
x = dataframe
y = dataframe["level_diff"]

# y.loc[y < 0] = -1
# y.loc[y > 0] = 1

del x["level_diff"]

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = tf.keras.models.Sequential(
    [
        Dense(310, input_dim=len(x.columns), kernel_initializer='normal', activation='relu'),
        Dense(310, input_dim=len(x.columns), kernel_initializer='normal', activation='relu'),
        Dense(310, input_dim=len(x.columns), kernel_initializer='normal', activation='relu'),
        Dense(310, input_dim=len(x.columns), kernel_initializer='normal', activation='relu'),
        Dense(155, input_dim=len(x.columns), kernel_initializer='normal', activation='relu'),
        Dense(155, input_dim=len(x.columns), kernel_initializer='normal', activation='relu'),
        Dense(77, input_dim=len(x.columns), kernel_initializer='normal', activation='relu'),
        Dense(1, activation='linear')
    ]
)

model.summary()
mse = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])
history = model.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=1, validation_split=0.2)

model.save('model1000epoch.h5')

plt.semilogy(history.history['mse'])
plt.semilogy(history.history['val_mse'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# print(model.evaluate(x_test, y_test))
