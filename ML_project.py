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

# Récupération des données


dataframe = pandas.read_csv('export.csv')
dataframe.sample(frac=1)
x = dataframe
y = dataframe["diff_level"]

del dataframe["diff_level"]

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = tf.keras.models.Sequential(
    [
        Dense(30, input_dim=3, kernel_initializer='normal', activation='relu'),
        Dense(1, activation='linear')
    ]
)

model.summary()
mse = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])
history = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.2)

plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()