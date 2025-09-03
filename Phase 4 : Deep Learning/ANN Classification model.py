# %% 
import pandas as pd
import numpy as np
import tensorflow as tf
# %% 
data = pd.read_csv('E:/taj/Downloads/Deep Learning A-Z/Part 1 - Artificial Neural Networks (ANN)/Churn_Modelling.csv')
X = data.iloc[:, 3:-1].values
y = data.iloc[:, -1].values
# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# %%
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=11, activation='relu'))
ann.add(tf.keras.layers.Dense(units=11, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# %%
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, batch_size=32, epochs=100)
# %%
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
# %%
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
# %%
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
# %%
