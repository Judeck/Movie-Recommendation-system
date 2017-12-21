import pickle
import pandas as pd
import numpy as np

# input prediction features
M = pickle.load(open("movie_f.dat", "rb"))

M = np.transpose(M)

movies_input = pd.read_csv('movie_features.csv', index_col='movie_id')

# input the movie ID to find the similar one
print("Hello! which movie do you like?")
movie_id = int(input())
print("Hello! How many movies do you want?")
number = int(input())

# input the movie information including id, title, genre
movie_inf = movies_input.loc[movie_id]

print("We are finding movies similar to this movie:")
print("Title of the Movie: {}".format(movie_inf.title))
print("Genre of the Movie: {}".format(movie_inf.genre))


# Applied LRMF to obtain features for target Movie 
current_movie_features = M[movie_id - 1]

print("The attributes for this movie are:")
print(current_movie_features)

# The progress to finding similar movies:
difference = M - current_movie_features

absolute_difference = np.abs(difference)

sum = np.sum(absolute_difference, axis=1)

movies_input['difference_score'] = sum

sort = movies_input.sort_values('difference_score')

# sort the difference between the target movie with other movies and get top number
print("The 3 similar movies are as below:")
print(sort[['title', 'difference_score']][0:number])
