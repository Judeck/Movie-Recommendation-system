import pickle
import pandas as pd
import numpy as np
import LRMF

# input movie data
dataset = pd.read_csv('movie.csv')

# transform ratings into matrix
ratings = pd.pivot_table(dataset, index='user_id', columns='movie_id', aggfunc=np.max)

# Apply LRMF to find movie features
U, M = LRMF.factorization(ratings.as_matrix(),features=15,regularization=0.1)

# Applied LRMF to the product of U and M
predicted = np.matmul(U, M)

# save the data as simple database
pickle.dump(U, open("user_f.dat", "wb"))
pickle.dump(M, open("movie_f.dat", "wb"))
pickle.dump(predicted, open("movie_rating.dat", "wb" ))

# load data from the database we saved before
U = pickle.load(open("user_f.dat", "rb"))
M = pickle.load(open("movie_f.dat", "rb"))
predicted = pickle.load(open("movie_rating.dat", "rb"))

# Input movie titles form  dataset
movies_input = pd.read_csv('movie_features.csv', index_col='movie_id')

# display the recommendation result
print("Hi! Which User you are?(1-100)")
user_id_to_search = int(input())
print("Hello! How many movies do you want?")
number = int(input())

print("These Movies are recommended:")

user_ratings = predicted[user_id_to_search - 1]
movies_input['rating'] = user_ratings
movies_input = movies_input.sort_values(by=['rating'], ascending=False)

print(movies_input[['title', 'genre', 'rating']].head(number))


### RMSE ###

# input training and testing data
traing = pd.read_csv('movie_training.csv')
testing = pd.read_csv('movie_testing.csv')

# transform input data to matrix
traning_ratings = pd.pivot_table(traing, index='user_id', columns='movie_id', aggfunc=np.max)
testing_ratings = pd.pivot_table(testing, index='user_id', columns='movie_id', aggfunc=np.max)


# Applied LMRF to U,M matrix to predict matrix
U, M = LRMF.factorization(traning_ratings.as_matrix(),features=30,regularization=2)

predicted = np.matmul(U, M)

# caculate training and testing dataset RMSE
traing_RMSE = LRMF.RMSE(traning_ratings.as_matrix(),predicted)
testing_RMSE = LRMF.RMSE(testing_ratings.as_matrix(),predicted)

print("RMSE of training data is: {}".format(traing_RMSE))
print("RMSE of testing data is: {}".format(testing_RMSE))
