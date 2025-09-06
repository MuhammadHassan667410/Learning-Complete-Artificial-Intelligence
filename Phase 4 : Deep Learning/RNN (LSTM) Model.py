# %% PART 1 DATA PREPROCESSING
# %% Importing Libraries

from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Loading Data

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# %% Feature Scaling

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# %% Creating a datastructure with 60 timesteps and 1 output

X_train = []
y_train = []
for i in range(120, 1258):
    X_train.append(training_set_scaled[i-120:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# %% Reshaping(Adding Additional Indicator)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# %% Part 2 BUILDING THE RNN
# %% Initializing the RNN

regressor = Sequential()

# %% Adding first LSTM Layer and some Dropout regularization

regressor.add(LSTM(units=60, return_sequences=True,
              input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# %% Adding Second LSTM Layer and some Dropout regularization

regressor.add(LSTM(units=60, return_sequences=True))
regressor.add(Dropout(0.2))

# %% Adding third LSTM Layer and some Dropout regularization

regressor.add(LSTM(units=60, return_sequences=True))
regressor.add(Dropout(0.2))

# %% Adding fourth LSTM layer and some Dropout regularization

regressor.add(LSTM(units=60))
regressor.add(Dropout(0.2))

# %% Adding the output layer

regressor.add(Dense(units=1))

# %% compiling the RNN

regressor.compile(optimizer='adam', loss='mean_squared_error')

# %% Fitting the RNN to training set

regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# %% PART 3 : MAKING THE PREDICTION AND VISUALIZING THE RESULTS
# %% Getting the real stock prices

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# %% Getting the predicted stock prices

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test)-120:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

# %% Creating 3D structure

X_test = []
for i in range(120, 140):
    X_test.append(inputs[i-120:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# %% Prediction

predicted_stock_prices = regressor.predict(X_test)
predicted_stock_prices = sc.inverse_transform(predicted_stock_prices)

# %% Visualizing the result

plt.plot(real_stock_price, color='red', label = 'Real Google Stock Prices')
plt.plot(predicted_stock_prices, color='blue', label='Predicted Google Stock Prices')
plt.title('GOOGLE  STOCK PRICE PREDICTION')
plt.xlabel('Time')
plt.ylabel('Google, Stock Prices')
plt.legend()
plt.show()

# %% Evaluating

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_prices))
print(rmse)
