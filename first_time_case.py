import numpy as np
import pickle
import pandas as pd
import LRMF

# normalize the data to avoid empty user value
def normalize(ratings):
    avg = np.nanmean(ratings, axis=0)
    return ratings - avg, avg

# input movie rate data
dataset = pd.read_csv('movie.csv')

# transform ratings into matrix
ratings = pd.pivot_table(dataset, index='user_id', columns='movie_id', aggfunc=np.max)

# Apply LRMF to find the movie features after normalizing data
normalize, avag = normalize(ratings.as_matrix())
U, M = LRMF.factorization(normalize,features=11,regularization=1.1)

# Applied LRMF to the product of U and M
predicted = np.matmul(U, M)
predicted = predicted + avag

# process the mean data for the first time user
pickle.dump(avag, open("avg.dat", "wb" ))
avag = pickle.load(open("avg.dat", "rb"))

# Load movie titles
movies = pd.read_csv('movie_features.csv', index_col='movie_id')

# provide the highest average rating movie to the first time user.

print("Movies we will recommend as below:")

movies['rating'] = avag
movies = movies.sort_values(by=['rating'], ascending=False)

# input the recommendation numbers
print("Hello! How many movies do you want?")
number = int(input())
print(movies[['title', 'genre', 'rating']].head(number))