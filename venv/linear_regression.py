import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

data = pd.read_csv("kc_house_data.csv")
space = data['sqft_living']
price = data['price']
print (price)
space = space / 100.0
price = price / 10000.0
space_arr = np.array(space).reshape(-1, 1)
price_arr = np.array(price).reshape(-1, 1)

print (space_arr[0])
print (price_arr.shape)
print (price_arr[0])

from sklearn.model_selection import train_test_split
space_train, space_test, price_train, price_test = train_test_split(space_arr,price_arr,test_size=0.4,train_size=0.6)
# min-max normalization
def MaxMinNormalization(x):
    """[0,1] normaliaztion"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def linear_regression(space_data, price_data, learning_rate):
    # learning_rate = 0.001
    initial_b = 0
    initial_m = 0
    # number of iterative
    num_iter = 300

    [b, m] = optimizer(space_data, price_data, initial_b, initial_m, learning_rate, num_iter)
#     plot the data here
    print (b, m)
    return b, m

def optimizer(space_data, price_data, initial_b, initial_m, learning_rate, num_iter):
    b = initial_b
    m = initial_m

    for i in range(num_iter):
        b, m = compute_gradient(b, m, space_data, price_data, learning_rate)

        if i % 10 == 0:
            print ('Iter: %s'%i, 'error: %s'%compute_error(b, m, space_data, price_data))

    return [b, m]

# gradient descent function
def compute_gradient(b_cur, m_cur, space_data, price_data, learning_rate):
    b_gradient = 0
    m_gradient = 0

    N = float(len(space_data))

    for i in range(0, len(space_data)):
        x = space_data[i]
        y = price_data[i]
        b_gradient += -(2 / N) * (y - ((m_cur * x) + b_cur))
        m_gradient += -(2 / N) * x * (y - ((m_cur * x) + b_cur))

    b_next = b_cur + (-learning_rate * b_gradient)
    m_next = m_cur + (-learning_rate * m_gradient)
    return (b_next, m_next)
# cost function
def compute_error(b, m, space_data, price_data):
    for i in range(len(space_data)):
        x = space_data[i]
        y = price_data[i]
        totalError = (y - m * x - b) ** 2
    totalError = np.sum(totalError, axis=0)
    return totalError/len(space_data)

if __name__ == '__main__':
    # training result
    start = time.thread_time()
    b, m = linear_regression(space_train, price_train, 0.0005)
    end = time.thread_time()
    print ('Time used: {}'.format(end - start))
    # visualize the test results
    plt.scatter(space_test, price_test, color='green')
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = b + m * x_vals
    plt.plot(x_vals, y_vals, '--', color='blue')
    plt.title("Visuals for Test DataSet")
    plt.xlabel("Space")
    plt.ylabel("Price")
    plt.show()
