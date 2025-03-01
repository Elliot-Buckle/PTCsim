import numpy as np

def first_derivative(array:np.ndarray, step:float):
    gradient = (np.roll(array, -1) - np.roll(array, 1))/(2*step)
    gradient[0] = (-3/2 * array[0] + 2 * array[1] - 1/2 * array[2])/step
    gradient[-1] = (3/2 * array[-1] - 2 * array[-2] + 1/2*array[-3])/step
    # gradient[0] = (array[1] - array[0])/step
    # gradient[-1]  = (array[-1] - array[-2])/step
    return gradient

def first_derivative_2(array:np.ndarray, step:float):
    # Central
    gradient = (1/12 * np.roll(array, 2) - 2/3*np.roll(array, 1) + 2/3*np.roll(array, -1) - 1/12*np.roll(array, -2))/step
    
    # Forward
    gradient[0] = (-25/12*array[0] + 4 * array[1] - 3 * array[2] + 4/3*array[3] - 1/4 * array[4])/step
    gradient[1] = (-25/12*array[1] + 4 * array[2] - 3 * array[3] + 4/3*array[4] - 1/4 * array[5])/step
    
    # Backward
    gradient[-1] = (25/12 * array[-1] - 4 * array[-2] + 3 * array[-3] - 4/3 * array[-4] + 1/4 * array[-5])/step
    gradient[-2] = (25/12 * array[-2] - 4 * array[-3] + 3 * array[-4] - 4/3 * array[-5] + 1/4 * array[-6])/step
    
    return gradient

def exponential_approximation(initial_value, drag_term, driving_term, tstep):
    return np.where(drag_term == 0, driving_term*tstep + initial_value, ((drag_term * initial_value + driving_term)*np.exp(drag_term*tstep) - driving_term)/drag_term)

if __name__ == "__main__":
    print(exponential_approximation(0, 0, 1, 0))