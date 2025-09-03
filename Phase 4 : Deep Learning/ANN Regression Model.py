# %%
import numpy as np
import pandas as pd
import tensorflow as tf
# %%
dataset = pd.read_excel('Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# %% Layers
ann = tf.keras.models.Sequential()
# First hidden Layer
ann.add(tf.keras.layers.Dense(units = 11, activation = 'relu'))
# Second Hidden Layer
ann.add(tf.keras.layers.Dense(units = 11, activation = 'relu'))
# Output layer
ann.add(tf.keras.layers.Dense(units=1))
# Compiling the layer
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Training the ann model
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
# %%
# Predicting and evaluating the model
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))