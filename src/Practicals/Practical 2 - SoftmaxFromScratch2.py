import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Load the data
train = pd.read_csv("data/MNISTKaggleData/train.csv")

def plot_digit(index):
    image = train.iloc[index, 1:]
    digit_label = train.iloc[index, 0]
    image_reshape = image.values.reshape(28, 28)
    plt.imshow(image_reshape, cmap='Greys')
    plt.title('Digit Label: {}'.format(digit_label))
    plt.show()


plot_digit(20)


def one_hot_array(Y):
    b = np.zeros((Y.size, Y.max()+1))
    b[np.arange(Y.size), Y] = 1
    return b.T


X = train.iloc[:, 1:].values.T
Y = train.iloc[:, 0]
Y_onehot = one_hot_array(Y.values)
print("Shape of X is: ", str(X.shape))
print("Shape of Y is: ", str(Y_onehot.shape))

X_train = X[:, 0:5000]
X_test = X[:, 5000:10000]
Y_train = Y_onehot[:, 0:5000]
Y_test = Y_onehot[:, 5000:10000]
print("Shape of X_train is: ", str(X_train.shape))
print("Shape of X_test is: ", str(X_test.shape))
print("Shape of Y_train is: ", str(Y_train.shape))
print("Shape of Y_test is: ", str(Y_test.shape))

# Initialize parameters


def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))

        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


parameters = initialize_parameters_deep([784, 4, 3, 2, 10])
print(parameters.keys())

# Forward propagation


def softmax(Z):
    t = np.exp(Z)
    t = t / t.sum(axis=0, keepdims=True)
    return t


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    return A


def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z = np.dot(W, A_prev) + b
        A = sigmoid(Z)

    elif activation == "relu":
        Z = np.dot(W, A_prev) + b
        A = relu(Z)

    elif activation == "softmax":
        Z = np.dot(W, A_prev) + b
        A = softmax(Z)

    assert(Z.shape == (W.shape[0], A.shape[1]))
    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    # Cache the values for backpropagation
    cache = (A_prev, W, b, Z)

    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2 # number of layers in the neural network

    # Hidden layers 1 to L-1 will be ReLU
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)

    # Output layer will be softmax
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="softmax")
    caches.append(cache)

    assert(AL.shape == (10, X.shape[1]))

    return AL, caches

# Probabilities without training the model

AL, caches = L_model_forward(X_train, parameters)
print(pd.DataFrame(AL[:, 0:5]))

# Compute cost

def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = -1/m * np.sum(np.multiply(Y, np.log(AL)))
    cost = np.squeeze(cost) # To coerce data from [[17]] into 17
    assert(cost.shape == ())

    return cost

print(compute_cost(AL, Y_train))

# Backward propagation
def linear_backward(dZ, A_prev, W, b):
    m = A_prev.shape[1]
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)

    return dA_prev, dW, db


def relu_backward(dZ, cache):
    A_prev, W, b, Z = cache

    # Compute dZ
    dZ = np.array(dZ, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)

    # Compute dA_prev, dW, db
    dA_prev, dW, db = linear_backward(dZ, A_prev, W, b)
    return dA_prev, dW, db


def softmax_backward(AL, Y, cache):
    A_prev, W, b, Z = cache

    # Compute dZ
    dZ = AL - Y

    # Compute dA_prev, dW, db
    dA_prev, dW, db = linear_backward(dZ, A_prev, W, b)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # To match the shape of AL

    # Backpropagation at layer L-1
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = softmax_backward(AL, Y, current_cache)

    # Backpropagation from layer L-2 to 1
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = relu_backward(grads["dA" + str(l+1)], current_cache)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads


grads = L_model_backward(AL, Y_train, caches)
print(grads.keys())

# Update parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters


parameters = update_parameters(parameters, grads, learning_rate=0.0075)
AL, caches = L_model_forward(X_train, parameters)
print(pd.DataFrame(AL[:, 0:5]))

# Training the model
def L_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []

    # Initialize parameters
    parameters = initialize_parameters_deep(layer_dims)

    # Iterative loops for gradient descent
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i%100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i%10 == 0:
            costs.append(cost)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

# Prediction and accuracy
def predict(X, parameters):
    # Forward propagation
    probabilites, caches = L_model_forward(X, parameters)

    # Calculate predicitons
    predictions = (probabilites == np.amax(probabilites, axis=0, keepdims=True))
    predictions = predictions.astype(float)

    return predictions, probabilites

def evaluate_prediction(predictions, Y):
    m = Y.shape[1]
    predictions_class = predictions.argmax(axis=0).reshape(1, m)
    Y_class = Y.argmax(axis=0).reshape(1, m)

    return np.sum((predictions_class == Y_class) / (m))

layers_dims = [784, 10, 10]
parameters = L_layer_model(X_train, Y_train, layers_dims, learning_rate=0.001, num_iterations=500, print_cost=True)

pred_train, probs_train = predict(X_train, parameters)
print("Train set error is: " + str(evaluate_prediction(pred_train, Y_train)))
pred_test, probs_test = predict(X_test, parameters)
print("Test set error is: " + str(evaluate_prediction(pred_test, Y_test)))

# Visualize the prediction
def plot_predicted_images(images, true_labels, predicted_labels, indices):
    # Plot the images, along with their true and predicted labels
    fig, axes = plt.subplots(1, len(indices), figsize=(10, 3))
    for idx, image, true_label, pred_label, ax in zip(indices, images[indices], true_labels[indices], predicted_labels[indices], axes):
        ax.imshow(image.reshape(28,28), cmap='Greys')
        ax.axis('off') # Hide grid lines
        ax.set_title(f'Idx: {idx}\nTrue: {true_label}, Pred: {pred_label}')
    plt.show()


indices = [11, 33, 56, 67]

# Compute the predicted labels for the test set
_, test_probs = predict(X_test, parameters)
test_preds = np.argmax(test_probs, axis=0)

# Get the original labels for the test set, not one-hot encoded
Y_test_orig = Y.values[5000:10000]

# Convert the test set images from column vectors to matrices (28x28)
X_test_images = X_test.T.reshape(-1, 28, 28)

plot_predicted_images(X_test_images, Y_test_orig, test_preds, indices)