#%% Part 1 : Building SOM Unsupervised model
# %% Imorting Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# %% Importing Dataset

Dataset = pd.read_csv('E:/taj/Downloads/Deep Learning A-Z/Part 4 - Self Organizing Maps (SOM)/Credit_Card_Applications.csv')
X = Dataset.iloc[:, :-1].values
y = Dataset.iloc[:, -1].values

# %% Feature sacaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

# %% Training SOM

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration=100)

# %% Visualizing the result

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
# %% Finding the frauds

mappings = som.win_map(X)

# get SOM distance map
dist_map = som.distance_map().T
threshold = dist_map.mean() + dist_map.std()  # suspicious threshold

# find suspicious nodes
suspect_coords = [(i, j) for i in range(10) for j in range(10) if dist_map[i, j] > threshold]

# safely collect frauds
frauds_list = [mappings[coord] for coord in suspect_coords if coord in mappings and len(mappings[coord]) > 0]

if frauds_list:  # only concatenate if not empty
    frauds = np.concatenate(frauds_list, axis=0)
    frauds = sc.inverse_transform(frauds)
else:
    frauds = np.array([])  # no suspicious cases found

# %% PART 2 : Going from unsupervised to superviesd Deep Learning
# %% Creating the matrix of features

customers = Dataset.iloc[:, 1:].values

# %% Creating the dependent variable

is_fraud = np.zeros(len(Dataset))
for i in range(len(Dataset)):
    if Dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
        
# %% Training ANN
# %% Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)
# %% Adding Layers

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=10, activation='relu', input_dim=15))
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# %% Compiling and fitting

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
ann.fit(customers, is_fraud, batch_size=1, epochs=50)

# %% Predicting the probabilities of frauds

y_pred = ann.predict(customers)

# %% Concatenation

y_pred = np.concatenate([Dataset.iloc[:, 0:1].values ,y_pred], axis=1)

# %% Sorting

y_pred = y_pred[y_pred[:, 1].argsort()]