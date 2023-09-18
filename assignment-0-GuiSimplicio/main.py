import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn import linear_model
from sklearn.datasets import make_blobs


def hello_world():

    # TODO: Check in the tests what the function should return and modify the return statement accordingly

    return "Hello World!"


def add(x, y):
    """
    Add the two input numbers.
    """

    # TODO: Insert your code here

    #Could check if input (x,y) is both numbers, but with the test done it is not necessary :)

    return x+y


def torch_version():
    """
    Return the version of torch that is loaded.
    """

    # TODO: Insert your code here, check the [pytorch documentation](https://pytorch.org/docs/stable/index.html) for help.
    number_of_version=torch.__version__.split('+',1)[0] #return of torch.__version__ is "number of version + something", we just want the number!  
    return number_of_version


def to_numpy(x):
    """
    Convert the input list to a numpy array.
    """

    # TODO: Insert your code here

    return np.array(x)


def simple_regression():
    """
    This function trains a linear regression model a very artifical dataset.

    It should return the estimated coefficients of the model.
    Check the sklearn documentation for help.
    """
    x = [[0, 0], [1, 1], [2, 2]] # data
    y = [0, 1, 2] # labels

    reg = linear_model.LinearRegression()
    reg.fit(x, y)

    # TODO Insert your code here

    return reg.coef_

def plot_gaussian_data():
    """
    This function should plot some data.

    It should look like shown in the README.md, check the matplotlib documentation for help.
    """
    
    X, y_true = make_blobs(n_samples=400, centers=4,
                        cluster_std=0.60, random_state=0)
    X = X[:, ::-1]


    # TODO Insert your code here
    plt.scatter(X[:, 0], X[:, 1],c=y_true)
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.savefig("results/scattered_data_example.png")
    plt.show()

   


if __name__ == "__main__":

    """ 
    This is the main function.

    It is executed when you run the script from the command line.
    'conda activate ml4phys'
    'python hello.py'

    You can write some code to use your funtions here.
    """

    plot_gaussian_data()
