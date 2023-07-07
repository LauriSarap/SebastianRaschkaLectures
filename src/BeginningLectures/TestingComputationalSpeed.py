import numpy as np
import timeit

x0, x1, x2 = 1., 2., 3.
bias, w1, w2 = 0.1, 0.3, 0.5

x = [x0, x1, x2]
w = [bias, w1, w2]

x_vec, w_vec = np.array(x), np.array(w)

z = x_vec.dot(w_vec)


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def forloop(x, w):
    z = 0.
    for i in range(len(x)):
        z += x[i] * w[i]
    return z


def listcomprehension(x, w):
    return sum(x_i * w_i for x_i, w_i in zip(x, w))


def vectorized(x, w):
    return x_vec.dot(w_vec)


x, w = np.random.rand(100000), np.random.rand(100000)

# Try out the forloop speed
wrapped = wrapper(forloop, x, w)
print(timeit.timeit(wrapped, number=10))

# Try out the list comprehension speed
wrapped = wrapper(listcomprehension, x, w)
print(timeit.timeit(wrapped, number=10))

# Try out the vectorized speed
wrapped = wrapper(vectorized, x, w)
print(timeit.timeit(wrapped, number=10))

