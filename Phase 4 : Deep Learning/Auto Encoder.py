# %% Importing the libraries

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# %% Importing the dataset

movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# %% Preparing the training set and the test set

training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# %% Getting the number of users and movies

nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# %% Converting the data into an array with users in lines and movies in columns

def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# %% Converting the data into Torch tensors

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# %% Creating the Architecture of the Nueral Network

class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# %% Training the SAE

nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
# %% Testing the SAE

test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))

# %% After training + testing the SAE

# Function to recommend Top-N movies for a given user
def recommend_movies(user_id, model, training_set, movies, N=5):
    # Put user ratings into model
    input = Variable(training_set[user_id-1]).unsqueeze(0)  # user_id starts at 1
    output = model(input)

    # Detach tensor to numpy
    predicted_ratings = output.data.numpy().flatten()

    # Mask already rated movies (don’t recommend what user has already seen)
    rated_movies = training_set[user_id-1].numpy() > 0
    predicted_ratings[rated_movies] = -1  # exclude seen movies

    # Get top-N recommendations
    top_indices = predicted_ratings.argsort()[-N:][::-1]  # highest to lowest
    top_movie_ids = top_indices + 1  # movie IDs are 1-based in MovieLens

    # Map to movie titles
    recommended_titles = movies.iloc[top_movie_ids - 1, 1].values
    return list(zip(recommended_titles, predicted_ratings[top_indices]))

# Example: Recommend top 5 movies for user 1
user_id = 1
recommendations = recommend_movies(user_id, sae, training_set, movies, N=5)

print(f"\nTop 5 movie recommendations for User {user_id}:")
for title, score in recommendations:
    print(f"{title} (Predicted score: {score:.3f})")
