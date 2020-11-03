import numpy as np
import matplotlib.pyplot as plt

def cross_entropy_error(x):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
