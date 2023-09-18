from unittest import result
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import sklearn.linear_model as linear_model
from sklearn.metrics import mean_squared_error


def load_and_normalize_data():
    # load the numpy arrays inputs and labels from the data folder
    # TODO
    X, y = np.loadtxt("data/inputs.txt"), np.loadtxt("data/labels.txt")

    # normalize the target y
    # TODO
    y = y / (np.std(y)) 

    return X, y


def data_summary(X, y):

    # return several statistics of the data
    # TODO
    X_mean = np.mean(X)
    X_std = np.std(X)
    y_mean = np.mean(y)
    y_std = np.std(y)
    X_min = np.min(X)
    X_max = np.max(X)

    return {'X_mean': X_mean,
            'X_std': X_std, 
            'X_min': X_min, 
            'X_max': X_max, 
            'y_mean': y_mean, 
            'y_std': y_std}


def data_split(X, y):
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=0.25, random_state=4)

    return X_train, X_test, y_train, y_test, X_validation, y_validation


def fit_linear_regression(X, y, lmbda=0.0, regularization=None):
    """
    Fit a ridge regression model to the data, with regularization parameter lmbda and a given
    regularization method.
    If the selected regularization method is None, fit a linear regression model without a regularizer.

    !! Do not fit the intersept in all cases.

    y = wx+c

    X: 2D numpy array of shape (n_samples, n_features)
    y: 1D numpy array of shape (n_samples,)
    lmbda: float, regularization parameter
    regularization: string, 'ridge' or 'lasso' or None

    Returns: The coefficients and intercept of the fitted model.
    """

    # TODO: use the sklearn linear_model module

    if regularization==None:
        lr=LinearRegression(fit_intercept=False).fit(X, y)
    elif regularization=="ridge":
        lr = linear_model.Ridge(alpha = lmbda,fit_intercept=False).fit(X, y) 
    elif regularization=="lasso":
        lr = linear_model.Lasso(alpha = lmbda, fit_intercept=False).fit(X, y)
    else:
        print("Regularization method invalid. \n")
        return 

    w = lr.coef_
    c = lr.intercept_

    return w, c


def predict(X, w, c):
    """
    Return a linear model prediction for the data X.

    X: 2D numpy array of shape (n_samples, n_features) data
    w: 1D numpy array of shape (n_features,) coefficients
    c: float intercept

    Returns: 1D numpy array of shape (n_samples,)
    """
    # TODO
    y_pred = X @ w + c 
    return y_pred


def mse(y_pred, y):
    """
    Return the mean squared error between the predictions and the true labels.

    y_pred: 1D numpy array of shape (n_samples,)
    y: 1D numpy array of shape (n_samples,)

    Returns: float
    """

    # TODO
    MSE = mean_squared_error(y, y_pred) 

    return MSE



def fit_predict_test(X_train, y_train, X_test, y_test, lmbda=0.0, regularization=None):
    """
    Fit a linear regression model, possibly with L2 regularization, to the training data.
    Record the training and testing MSEs.
    Use methods you wrote before

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    lmbda: float, regularization parameter

    Returns: The coefficients and intercept of the fitted model, the training and testing MSEs in a dictionary.
    """

    w, c = fit_linear_regression(X_train, y_train, lmbda, regularization) 
    y_train_pred=predict(X_train,w,c)

    results = {
        'mse_train':mse(y_train_pred,y_train),
        'mse_test':mse(X_test @ w, y_test),
        'lmbda':lmbda,
        'w':w,
        'c':c,
    }

    return results


def plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas, lmbda=0.0, regularization=None, filename=None):
    """
    Plot the training and testing MSEs against the regularization parameter alpha.
    Use the functions you just wrote.

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    alphas: list of values, the dataset percentage to be checked (alpha=n/d)
    lmbda: float, regularization parameter
    regularization: string, 'ridge' or 'lasso' or None
    filename: string, name to save the plot

    Returns: None
    """
    
    # TODO: You might want to use the pandas dataframe to store the results
    # your code goes here
    
    MSE_train = []
    MSE_test = []
    #alpha = n/d => n = alpha * d

    d=X_train.shape[1] 
    n_list=(np.array(alphas)*d).tolist()


    for n in n_list:
        
        results=fit_predict_test(X_train[:int(n)], y_train[:int(n)], X_test, y_test, lmbda, regularization)
        # TODO
        MSE_test.append(results["mse_test"]) 
        MSE_train.append(results["mse_train"]) 


    plt.plot(alphas,MSE_train,label='train')
    plt.plot(alphas,MSE_test,label='test')

    
    plt.xlabel(r'$\alpha = \frac {n}{d}$ ')
    plt.ylabel('mse')
    plt.legend()
    plt.title("Plot of data vs mean squared error")
    plt.savefig(f'results/{filename}.png')
    plt.clf()


def plot_regularizer_vs_coefficients(X_train, y_train, X_test, y_test, lmbdas,  plot_coefs, regularization='ridge', filename=None):
    """
    Plot the coefficients of the fitted model against the regularization parameter alpha.

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    lmbdas: list of values, the regularization parameter
    plot_coefs: list of integers, the coefficients of w to be plotted
    regularization: string, 'ridge' or 'lasso' or None
    filename: string, name to save the plot

    Returns: None
    """

    # TODO: You might want to use the pandas dataframe to store the results
    # your code goes here

    w=[]

    for lmbda in lmbdas: # loop over the regularizer parameters
        res={}
        res["lambda"] = lmbda # populate the first colum of the dictionary with the lambda values
        
        W,c=fit_linear_regression(X_train, y_train, lmbda, regularization)
        
        for i, c in [(i, W[i]) for i in plot_coefs]:
            res[f"W[{i}]"] = c
        w.append(res)
    
    df=pd.DataFrame(w) # translates the dictionary into a pandas dataframe
    
    # process the result data in a pandas dataframe
    df.set_index('lambda',inplace=True)
    df.plot()
    plt.legend(loc='lower right', title='coef')
    plt.ylabel('coefficient value')
    plt.title("Regularization Strength vs Coefficient Weights")
    plt.xlabel(r"$\lambda$")
    plt.savefig(f'results/{filename}.png')
    plt.clf()

def add_poly_features(X):
    """
    Add squared features to the data X and return a new vector X_poly that contains
    X_poly[i,j]   = X[i,j]
    X_poly[i,2*j] = X[i,j]^2

    X: 2D numpy array of shape (n_samples, n_features)

    Returns: 2D numpy array of shape (n_samples, 2 * n_features ) with the normal and squared features
    """

    # TODO

    X_poly=np.zeros([X.shape[0],X.shape[1]*2]) #fills array  with dimensions: (n_samples, 2 * n_features )
    X_poly[:,:X.shape[1]]= X                   #Populates (n_samples, n_features ) with X values
    X_poly[:,X.shape[1]:]= X**2                #Populates (n_samples, n_features + d  ) with X^2 values

    return X_poly

def optimize_lambda(X_train,X_test,X_validation,y_train,y_test,y_validation, lmbdas,filename):

    regularization='ridge'

    # TODO: Experiment code goes here


    

    lmb_vs_MSE=[]

    for lmbda in lmbdas:
        res={}
        res["lambda"]= lmbda # populate the first colum of the dictionary with the lambda values
        aux=fit_predict_test(X_train, y_train, X_validation, y_validation, lmbda, regularization)
        res["MSE_train"]=aux["mse_train"] # Note that you should use mse_test!!
        lmb_vs_MSE.append(res)

    df=pd.DataFrame(lmb_vs_MSE)
    
    best_train_mse= df["MSE_train"].min()
    best_lmbda= df["lambda"][df["MSE_train"]==best_train_mse].item()
    best_test_mse=fit_predict_test(X_train, y_train, X_test, y_test, best_lmbda, regularization)["mse_test"].item()

    # TODO: Plotting code goes here
    df.plot(x="lambda",y="MSE_train")
    #plt.axvline(best_lmbda, color='red', label=r'Best $\lambda$')
    plt.legend()
    plt.title(r"Optimal $\lambda$")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("MSE of train data set")
    plt.savefig(f'results/{filename}.png')
    plt.clf()

    return best_test_mse, best_lmbda
  

if __name__ == "__main__":

    """ 
    !!!!! DO NOT CHANGE THE NAME OF THE PLOT FILES !!!!!
    They need to show in the Readme.md when you submit your code.

    It is executed when you run the script from the command line.
    'conda activate ml4phys-a1'
    'python main.py'

    This already includes the code for generating all the relevant plots.
    You need to fill in the ...
    """

    ## Exercise 1.
    # Load the data
    X, y = load_and_normalize_data()
    print("Successfully loaded and normalized data.")
    print(data_summary(X, y))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, X_validation, y_validation = data_split(X, y)

    n, d = X_train.shape

    ## Exercise 6.
    print('Find the optimal parameters for the Ridge regression...')
    lmbdas = np.arange(0,10000.1,0.1).tolist()
    n_ = 80
    lmbda, gen_error = optimize_lambda(X_train[:n_],X_test,X_validation,y_train[:n_],y_test,y_validation, lmbdas,filename='train_optimal_lambda_ridge_n80_std=1')
    print(n_, lmbda, gen_error)
    n_ = 150
    lmbda, gen_error = optimize_lambda(X_train[:n_],X_test,X_validation,y_train[:n_],y_test,y_validation, lmbdas,filename='train_optimal_lambda_ridge_n150_std=1')
    print(n_, lmbda, gen_error)
    print()


    print('Done. All results saved in the results folder.')

