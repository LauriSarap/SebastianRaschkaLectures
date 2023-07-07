import torch


def to_onehot(y, num_classes):
    y_onehot = torch.zeros(y.shape[0], num_classes)
    y_onehot.scatter_(1, y.view(-1, 1).long(), 1).float()
    return y_onehot


y = torch.tensor([0, 1, 2, 2])

y_enc = to_onehot(y, num_classes=3)

print('one-hot encoding:\n', y_enc)

Z = torch.tensor([[-0.3, -0.5, -0.5],
                  [-0.4, -0.1, -0.5],
                  [-0.3, -0.94, -0.5],
                  [-0.99, -0.88, -0.5]])


def softmax(Z):
    return (torch.exp(Z.t()) / torch.sum(torch.exp(Z), dim=1)).t()


smax = softmax(Z)
print('softmax:\n', smax)


def to_classlabel(z):
    return torch.argmax(z, dim=1)


print('Predicted class labels: ', to_classlabel(smax))
print('True class labels: ', to_classlabel(y_enc))


def cross_entropy(softmax, y_target):
    return - torch.sum(torch.log(softmax) * (y_target), dim=1)


xent = cross_entropy(smax, y_enc)
print('Cross Entropy: ', xent)

