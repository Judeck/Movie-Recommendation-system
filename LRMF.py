import numpy as np
from scipy.optimize import fmin_cg


# Using root mean square error to measure the accuracy
def RMSE(real_data, predicted_data):
    return np.sqrt(np.nanmean(np.square(real_data - predicted_data)))

# Using iteration algorithm to let the cost close to 0 to use LRMF
def cost(X, *args):
    
    users, movies, features, ratings, value, regularization = args

    U = X[0:(users * features)].reshape(users, features)
    M = X[(users * features):].reshape(movies, features)
    M = M.T


    return (np.sum(np.square(value * (np.dot(U, M) - ratings))) / 2) \
           + ((regularization / 2.0) * np.sum(np.square(M.T))) +\
           ((regularization / 2.0) * np.sum(np.square(U)))

# caculate the matrix U and M gradients to convergence the prediction
def grad(X, *args):

    users, movies, features, ratings, value, regularization = args

    U = X[0:(users * features)].reshape(users, features)
    M = X[(users * features):].reshape(movies, features)
    M = M.T

    U_grad = np.dot((value * (np.dot(U, M) - ratings)), M.T) + (regularization * U)
    M_grad = np.dot((value * (np.dot(U, M) - ratings)).T, U) + (regularization * M.T)

    return np.append(U_grad.ravel(), M_grad.ravel())

# Major function for LRMF to factarize low-rank matrix
def factorization(ratings, value=None, features=15, regularization=0.01):

    users, movies = ratings.shape

    if value is None:
        value = np.invert(np.isnan(ratings))

    ratings = np.nan_to_num(ratings)

    np.random.seed(0)
    U = np.random.randn(users, features)
    M = np.random.randn(movies, features)

    input = np.append(U.ravel(), M.ravel())

    args = (users, movies, features, ratings, value, regularization)

    X = fmin_cg(cost, input, fprime=grad, args=args, maxiter=9000)

    fU = X[0:(users * features)].reshape(users, features)
    fM = X[(users * features):].reshape(movies, features)

    return fU, fM.T

