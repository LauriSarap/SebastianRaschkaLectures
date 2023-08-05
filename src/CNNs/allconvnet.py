import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from helper_evaluation import set_all_seeds, set_deterministic, compute_confusion_matrix
from helper_train import train_model
from helper_plotting import plot_training_loss, plot_accuracy, show_examples, plot_confusion_matrix
from helper_dataset import get_dataloaders_cifar10, UnNormalize


def main():
    # Settings
    RANDOM_SEED = 123
    BATCH_SIZE = 256
    NUM_EPOCHS = 50
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_all_seeds(RANDOM_SEED)


    # CIFAR-10 DATASET
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.RandomCrop((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                          ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.CenterCrop((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])




    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=BATCH_SIZE,
        validation_fraction=0.1,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        num_workers=2)

    # Checking the dataset
    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        print('Class labels of 10 examples:', labels[:10])
        break

    # Model
    class AllConvNet(torch.nn.Module):
        def __init__(self, num_classes):
            super().__init__()

            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=3,
                                out_channels=16,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1,
                                bias=False),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=16,
                                out_channels=16,
                                kernel_size=(3, 3),
                                stride=(2, 2),
                                padding=1,
                                bias=False),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=16,
                                out_channels=32,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1,
                                bias=False),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=32,
                                out_channels=32,
                                kernel_size=(3, 3),
                                stride=(2, 2),
                                padding=1,
                                bias=False),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=32,
                                out_channels=64,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1,
                                bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=64,
                                out_channels=64,
                                kernel_size=(3, 3),
                                stride=(2, 2),
                                padding=1,
                                bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=64,
                                out_channels=num_classes,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1,
                                bias=False),
                torch.nn.BatchNorm2d(10),
                torch.nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten()
            )

        def forward(self, x):
            x = self.net(x)
            return x

    model = AllConvNet(num_classes=10)
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, verbose=True)

    minibatch_loss_list, train_acc_list, valid_acc_list = train_model(
        model=model,
        num_epochs=NUM_EPOCHS,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        scheduler_on='valid_acc',
        logging_interval=100)

    plot_training_loss(minibatch_loss_list=minibatch_loss_list,
                       num_epochs=NUM_EPOCHS,
                       iter_per_epoch=len(train_loader),
                       results_dir=None,
                       averaging_iterations=200)

    plt.show()
    plot_accuracy(train_acc_list=train_acc_list,
                  valid_acc_list=valid_acc_list,
                  results_dir=None)

    plt.ylim([60, 100])
    plt.show()

if __name__ == "__main__":
    main()
