import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# Hyperparameters
num_epochs = 40
learning_rate = 0.1
num_inputs = 28*28
num_outputs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Loading the data and plotting

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # Transform for normalising the data

trainset = datasets.FashionMNIST('data/', download=True, train=True, transform=transform) # Downloading the training data
trainloader = DataLoader(trainset, batch_size=256, shuffle=True) # Loading the training data

testset = datasets.FashionMNIST('data/', download=True, train=False, transform=transform) # Downloading the test data
testloader = DataLoader(testset, batch_size=256, shuffle=True) # Loading the test data


def plot_loss(losses):
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


# The model

class SoftmaxRegression(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        out = self.linear(x)
        return out


model = SoftmaxRegression(num_inputs, num_outputs)
model.load_state_dict(torch.load('practical1.ckpt'))
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Training the model
def train_model(model, criterion, optimizer, train_loader, num_epochs):
    total_start_time = time.time()
    losses = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        losses.append(avg_loss)
        epoch_end_time = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {epoch_end_time - epoch_start_time:.2f} seconds, Accuracy: {accuracy:.2f}%')

    total_end_time = time.time()
    print(f'Total training time: {total_end_time - total_start_time:.2f} seconds')

    return losses


#losses = train_model(model, criterion, optimizer, trainloader, num_epochs)
#plot_loss(losses)
#torch.save(model.state_dict(), 'practical1.ckpt')

# Evaluating the model


def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            images = images.view(images.shape[0], -1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {}%'.format(100 * correct / total))


evaluate_model(model, testloader)

def plot_examples(model, test_loader):
    model.eval()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images[:5]
        labels = labels[:5]
        outputs = model(images.view(images.shape[0], -1).to(device))
        _, predictions = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        for i in range(5):
            plt.imshow(images[i].squeeze(), cmap='summer')
            plt.title(f'Real: {class_names[labels[i]]}, Predicted: {class_names[predictions[i]]}')
            plt.show()


plot_examples(model, testloader)
