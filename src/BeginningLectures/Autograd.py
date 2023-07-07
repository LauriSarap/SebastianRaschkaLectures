import torch
from torch.autograd import grad
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

# Import the data

df = pd.read_csv('data/iris.data', index_col=None, header=None)
df.columns = ['x1', 'x2', 'x3', 'x4', 'y']
df = df.iloc[50:150]
df['y'] = df['y'].apply(lambda x: 0 if x == 'Iris-versicolor' else 1)

# Assign features and target
X = torch.tensor(df[['x2', 'x3']].values, dtype=torch.float)
y = torch.tensor(df['y'].values, dtype=torch.int)

# Shuffling & train/test split

torch.manual_seed(123)
shuffle_idx = torch.randperm(y.size(0), dtype=torch.long)

X, y = X[shuffle_idx], y[shuffle_idx]

percent70 = int(shuffle_idx.size(0)*0.7)

X_train, X_test = X[shuffle_idx[:percent70]], X[shuffle_idx[percent70:]]
y_train, y_test = y[shuffle_idx[:percent70]], y[shuffle_idx[percent70:]]

# Normalize (mean zero, unit variance)

mu, sigma = X_train.mean(dim=0), X_train.std(dim=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

# Define the model using autograd


class Adaline2():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features, 1, dtype=torch.float, requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float, requires_grad=True)

    def forward(self, x):
        net_inputs = torch.add(torch.mm(x, self.weights), self.bias)
        activations = net_inputs
        return activations.view(-1)

# Training and evaluation wrappers


def loss_function(y_hat, y):
    return torch.mean((y_hat - y)**2)


def train(model, x, y, num_epochs, learning_rate=0.01, seed=123, minibatch_size=10):
    cost = []

    torch.manual_seed(seed)
    for e in range(num_epochs):
        # Shuffle epoch
        shuffle_idx = torch.randperm(y.size(0), dtype=torch.long)
        minibatches = torch.split(shuffle_idx, minibatch_size)

        for minibatch_idx in minibatches:
            # Compute outputs
            y_hat = model.forward(x[minibatch_idx])
            loss = loss_function(y_hat, y[minibatch_idx])

            # Compute gradients
            negative_grad_w = grad(loss, model.weights, retain_graph=True)[0] * (-1)
            negative_grad_b = grad(loss, model.bias)[0] * (-1)

            # Update weights
            model.weights = model.weights + learning_rate * negative_grad_w
            model.bias = model.bias + learning_rate * negative_grad_b

        # Logging
        with torch.no_grad():
            # Save memory by not saving computation graphs
            y_hat = model.forward(x)
            curr_loss = loss_function(y_hat, y)
            print('Epoch: %03d' % (e + 1), end="")
            print(' | MSE: %.5f' % curr_loss)
            cost.append(curr_loss)

    return cost


model = Adaline2(num_features=X_train.size(1))
cost = train(model, X_train, y_train.float(), num_epochs=20, learning_rate=0.01, seed=123, minibatch_size=10)

plt.plot(range(len(cost)), cost)
plt.ylabel('MSE Adaline 2')
plt.xlabel('Epoch')
plt.show()

# Evaluation
ones = torch.ones(y_train.size())
zeros = torch.zeros(y_train.size())
train_pred = model.forward(X_train)
train_acc = torch.mean((torch.where(train_pred > 0.5, ones, zeros).int() == y_train).float())

ones = torch.ones(y_test.size())
zeros = torch.zeros(y_test.size())
test_pred = model.forward(X_test)
test_acc = torch.mean((torch.where(test_pred > 0.5, ones, zeros).int() == y_test).float())

print('Train accuracy: %.2f%%' % (train_acc*100))
print('Test accuracy: %.2f%%' % (test_acc*100))

# Automatic adaline

class Adaline3(torch.nn.Module):
    def __init__(self, num_features):
        super(Adaline3, self).__init__()
        self.linear = torch.nn.Linear(num_features, 1)

        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()

    def forward(self, x):
        netinputs = self.linear(x)
        activations = netinputs
        return activations.view(-1)


def train2(model, x, y, num_epochs, learning_rate=0.01, seed=123, minibatch_size=10):
    cost = []

    torch.manual_seed(seed)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for e in range(num_epochs):
        # Shuffle epoch
        shuffle_idx = torch.randperm(y.size(0), dtype=torch.long)
        minibatches = torch.split(shuffle_idx, minibatch_size)

        for minibatch_idx in minibatches:
            # Compute outputs
            y_hat = model.forward(x[minibatch_idx])
            loss = F.mse_loss(y_hat, y[minibatch_idx])

            # Reset gradients from previous iteration
            optimizer.zero_grad()

            # Compute gradients
            loss.backward()

            # Update weights
            optimizer.step()


        # Logging
        with torch.no_grad():
            # Save memory by not saving computation graphs
            y_hat = model.forward(x)
            curr_loss = loss_function(y_hat, y)
            print('Epoch: %03d' % (e + 1), end="")
            print(' | MSE: %.5f' % curr_loss)
            cost.append(curr_loss)

    return cost


model = Adaline3(num_features=X_train.size(1))
cost = train2(model, X_train, y_train.float(), num_epochs=20, learning_rate=0.01, seed=123, minibatch_size=10)

plt.plot(range(len(cost)), cost)
plt.ylabel('MSE Adaline 3')
plt.xlabel('Epoch')
plt.show()

# Evaluation
ones = torch.ones(y_train.size())
zeros = torch.zeros(y_train.size())
train_pred = model.forward(X_train)
train_acc = torch.mean(
    (torch.where(train_pred > 0.5,
                 ones,
                 zeros).int() == y_train).float())

ones = torch.ones(y_test.size())
zeros = torch.zeros(y_test.size())
test_pred = model.forward(X_test)
test_acc = torch.mean(
    (torch.where(test_pred > 0.5,
                 ones,
                 zeros).int() == y_test).float())

print('Training Accuracy: %.2f' % (train_acc*100))
print('Test Accuracy: %.2f' % (test_acc*100))
