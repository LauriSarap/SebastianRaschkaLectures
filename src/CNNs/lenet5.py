import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from helper_evaluation import set_all_seeds, set_deterministic, compute_confusion_matrix
from helper_train import train_model
from helper_plotting import plot_training_loss, plot_accuracy, show_examples, plot_confusion_matrix
from helper_dataset import get_dataloaders_mnist

# Settings
RANDOM_SEED = 123
BATCH_SIZE = 256
NUM_EPOCHS = 5
DEVICE = torch.device('cpu')

set_all_seeds(RANDOM_SEED)
set_deterministic()

# MNIST dataset
resize_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))])

train_loader, valid_loader, test_loader = get_dataloaders_mnist(
    batch_size=BATCH_SIZE,
    validation_fraction=0.1,
    train_transforms=resize_transform,
    test_transforms=resize_transform)

# Checking the dataset
for images, labels in train_loader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    print('Class labels of 10 examples:', labels[:10])
    break


# Model
class LeNet5(torch.nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super().__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        # Feature extractor
        self.features = torch.nn.Sequential(
            # 1st conv layer
            torch.nn.Conv2d(in_channels, 6, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2),
            # 2nd conv layer
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        # MLP as a Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16 * 5 * 5, 120),
            torch.nn.Tanh(),
            torch.nn.Linear(120, 84),
            torch.nn.Tanh(),
            torch.nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x) # NCHW format - Batch, Channel, Height, Width
        x = torch.nn.Flatten()(x) # N*M Matrix, where M = C * H * W
        logits = self.classifier(x)
        return logits


model = LeNet5(num_classes=10, grayscale=True)
model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, verbose=True)

# Load the model if not training
#model.load_state_dict(torch.load('saved_data/lenet5_mnist.pt'))
#optimizer.load_state_dict(torch.load('saved_data/lenet5_mnist_optimizer.pt'))
#scheduler.load_state_dict(torch.load('saved_data/lenet5_mnist_scheduler.pt'))


minibatch_loss_list, train_acc_list, valid_acc_list = train_model(
    model=model,
    num_epochs=NUM_EPOCHS,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    device=DEVICE,
    scheduler=scheduler,
    scheduler_on='valid_acc',
    logging_interval=100)

# Plotting
plot_training_loss(minibatch_loss_list=minibatch_loss_list,
                   num_epochs=NUM_EPOCHS,
                   iter_per_epoch=len(train_loader),
                   results_dir=None,
                   averaging_iterations=100)
plt.show()
plot_accuracy(train_acc_list=train_acc_list, valid_acc_list=valid_acc_list, results_dir=None)
plt.ylim([80, 100])
plt.show()

show_examples(model=model, data_loader=test_loader)

# Saving
torch.save(model.state_dict(), 'saved_data/lenet5_mnist.pt')
torch.save(optimizer.state_dict(), 'saved_data/lenet5_mnist_optimizer.pt')
torch.save(scheduler.state_dict(), 'saved_data/lenet5_mnist_scheduler.pt')