import pandas as pd
import matplotlib.pyplot as plt
import torch

# Load data

df = pd.read_csv('data/linreg-data.csv', index_col=0)
print(df.tail())

# Assign features and target

X = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float)
y = torch.tensor(df['y'].values, dtype=torch.float)

# Shuffling & train/test split

torch.manual_seed(123)
shuffle_idx = torch.randperm(y.size(0), dtype=torch.long)

X, y = X[shuffle_idx], y[shuffle_idx]

percent70 = int(shuffle_idx.size(0) * 0.7)

X_train, X_test = X[shuffle_idx[:percent70]], X[shuffle_idx[percent70:]]
y_train, y_test = y[shuffle_idx[:percent70]], y[shuffle_idx[percent70:]]

# Normalize (mean zero, unit variance), standardization
mu, sigma = X_train.mean(dim=0), X_train.std(dim=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma


class LinearRegression():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features, 1, dtype=torch.float)
        self.bias = torch.zeros(1, dtype=torch.float)

    def activation_func(self, x):
        return x

    def forward(self, x):
        net_inputs = torch.add(torch.mm(x, self.weights), self.bias)
        activations = self.activation_func(net_inputs)
        return activations.view(-1)

    def backward(self, x, yhat, y):

        # (yhat - y)^2
        # derivative: 2*(yhat - y)

        grad_loss_yhat = 2*(yhat - y)

        grad_yhat_weights = x
        grad_yhat_bias = 1.

        # Chain rule: inner times outer
        grad_loss_weights = torch.mm(grad_yhat_weights.t(), grad_loss_yhat.view(-1, 1)) / y.size(0)
        grad_loss_bias = torch.sum(grad_yhat_bias * grad_loss_yhat) / y.size(0)

        # Return negative gradient
        return (-1)*grad_loss_weights, (-1)*grad_loss_bias


def loss(yhat, y):
    return torch.mean((yhat - y)**2)


def train(model, x, y, num_epochs, learning_rate=0.01):
    cost = []
    for e in range(num_epochs):

        # Compute outputs
        yhat = model.forward(x)

        # Compute gradients
        negative_grad_w, negative_grad_b = model.backward(x, yhat, y)

        # Update weights
        model.weights += learning_rate * negative_grad_w
        model.bias += learning_rate * negative_grad_b

        # Logging
        yhat = model.forward(x)
        curr_loss = loss(yhat, y)
        print('Epoch: %03d' % (e+1), end="")
        print(' | MSE: %.5f' % curr_loss)
        cost.append(curr_loss)

    return cost

# Train the model


model = LinearRegression(num_features=X_train.size(1))
cost = train(model, X_train, y_train, num_epochs=100, learning_rate=0.05)

# Plot the training progress

plt.plot(range(len(cost)), cost)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.show()

train_pred = model.forward(X_train)
test_pred = model.forward(X_test)

print('Train MSE: %.5f' % loss(train_pred, y_train))
print('Test MSE: %.5f' % loss(test_pred, y_test))

# Compare with analytical solution
print('Weights', model.weights)
print('Bias', model.bias)


def analytical_solution(x, y):
    Xb = torch.cat((torch.ones((x.size(0), 1)), x), dim=1)
    w = torch.zeros(x.size(1))
    z = torch.inverse(torch.matmul(Xb.t(), Xb))
    params = torch.matmul(z, torch.matmul(Xb.t(), y))
    b, w = torch.tensor([params[0]]), params[1:].view(x.size(1), 1)
    return w, b


w, b = analytical_solution(X_train, y_train)
print('Analytical solution weights:', w)
print('Analytical solution bias:', b)
