import numpy as np

def first_derivative(array:np.ndarray, step:float):
    gradient = (np.roll(array, -1) - np.roll(array, 1))/(2*step)
    gradient[0] = (-3/2 * array[0] + 2 * array[1] - 1/2 * array[2])/step
    gradient[-1] = (3/2 * array[-1] - 2 * array[-2] + 1/2*array[-3])/step
    return gradient