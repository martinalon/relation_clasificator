import random
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

absolute_path = os.path.abspath(os.path.dirname(__file__))

def y_functions(x):
    """
    This function creates compute one mathematical
    function of a list of numbers
    
    Parameters:
    x: a list of numbers
    
    Return:
    y: a lis of numbers computed by a random mathematical
       function 
    """
    list1 = [*range(0, 13)]
    n = random.choice(list1)
    if n == 0:
        y = np.array(x) + np.random.normal(loc=0.0, scale=1.0, size=len(x))
    elif n == 1:
        y = - np.array(x) + np.random.normal(loc=0.0, scale=1.0, size=len(x))
    elif n == 2:
        y = np.sin(x) + np.random.normal(loc=0.0, scale=0.2, size=len(x))
    elif n == 3:	
        y = np.cos(x) + np.random.normal(loc=0.0, scale=0.2, size=len(x))
    elif n == 4:
        y = np.sinh(x) + np.random.normal(loc=0.0, scale=1.0, size=len(x))
    elif n == 5:
        y = np.cosh(x) + np.random.normal(loc=0.0, scale=1.0, size=len(x))
    elif n == 6:
        y = np.tanh(x) + np.random.normal(loc=0.0, scale=0.2, size=len(x))
    elif n == 7:
        y = np.exp(x) + np.random.normal(loc=0.0, scale=1.0, size=len(x))
    elif n == 8:
        y = np.log(x) + np.random.normal(loc=0.0, scale=1.0, size=len(x))
    elif n == 9:
        y = np.array(x)**2 + np.random.normal(loc=0.0, scale=1.0, size=len(x))
    elif n == 10:
        y = np.array(x)**3 + np.random.normal(loc=0.0, scale=1.0, size=len(x))
    elif n == 11:
        y = np.sqrt(x)  + np.random.normal(loc=0.0, scale=1.0, size=len(x))
    elif n == 12:
        y = - np.sqrt(x)  - np.random.normal(loc=0.0, scale=1.0, size=len(x))
    return(y)


def y_no_relation(x):
    """
    This function creates random numbers
    and define an array wit the same lenght
    that x.
    """
    list1 = [*range(0, 4)]
    n = random.choice(list1)
    if n == 0:
        y = np.random.normal(loc=0.0, scale=1.0, size=len(x))
    elif n == 1:
        y1 = np.random.normal(loc=0.0, scale=.45, size=len(x))
        y2 = 3 + np.random.normal(loc=0.0, scale=.45, size=len(x))
        y = np.concatenate((y1, y2), axis=0)
        y = np.random.choice(y, size=len(x), replace=False)
    elif n == 2:
        y1 = np.random.normal(loc=0.0, scale=.45, size=len(x))
        y2 = 3 + np.random.normal(loc=0.0, scale=.45, size=len(x))
        y3 = 6 + np.random.normal(loc=0.0, scale=.45, size=len(x))
        y = np.concatenate((y1, y2), axis=0)
        y = np.concatenate((y, y3), axis=0)
        y = np.random.choice(y, size=len(x), replace=False)
    else:
        y1 = np.random.normal(loc=0.0, scale=.45, size=len(x))
        y2 = 3 + np.random.normal(loc=0.0, scale=.45, size=len(x))
        y3 = 6 + np.random.normal(loc=0.0, scale=.45, size=len(x))
        y4 = 9 + np.random.normal(loc=0.0, scale=.45, size=len(x))
        y = np.concatenate((y1, y2), axis=0)
        y = np.concatenate((y, y3), axis=0)
        y = np.concatenate((y, y4), axis=0)
        y = np.random.choice(y, size=len(x), replace=False)
    return(y)



for i in tqdm(range(0, 1000)):
    x = np.random.uniform(low=0.001, high=10, size=(200,))
    y = y_functions(x)
    y_np = y_no_relation(x)
    
    plt.scatter(x, y, c = "black")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(absolute_path + '/scatter_plots/alternative_test/relation/' + str(i) + '.jpg')
    plt.clf()
    
    plt.scatter(x, y_np, c = "black")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(absolute_path + '/scatter_plots/alternative_test/no_relation/' + str(i) + '.jpg')
    plt.clf()
   