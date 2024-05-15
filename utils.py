import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %matplotlib widget
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from public_tests import * 

tf.keras.backend.set_floatx('float64')
from utils import *

tf.autograph.set_verbosity(0)
#! ---------------------------------------------------------------------------------

""" generate a data set based on a x^2 with added noise """
def gen_data(m, seed=1, scale=0.7):
    c = 0
    x_train = np.linspace(0,49,m)
    np.random.seed(seed)
    y_ideal = x_train**2 + c
    y_train = y_ideal + scale * y_ideal*(np.random.sample((m,))-0.5)
    x_ideal = x_train #for redraw when new data included in X
    return x_train, y_train, x_ideal, y_ideal


""" Plot the training and test sets """
def plot_train_and_test_sets(x_ideal, y_ideal, X_train, X_test, y_train, y_test):
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax.plot(x_ideal, y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
    ax.set_title("Training, Test",fontsize = 14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.scatter(X_train, y_train, color = "red",           label="train")
    ax.scatter(X_test, y_test,   color = "blue",   label="test")
    ax.legend(loc='upper left')
    
     # Save the figure
    
    fig.savefig('images/train-test_plot.png')