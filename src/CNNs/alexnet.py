import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from helper_evaluation import set_all_seeds, set_deterministic, compute_confusion_matrix
from helper_train import train_model
from helper_plotting import plot_training_loss, plot_accuracy, show_examples, plot_confusion_matrix
from helper_dataset import get_dataloaders_cifar10, UnNormalize

# Settings

RANDOM_SEED = 123
BATCH_SIZE = 256
NUM_EPOCHS = 90
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

set_all_seeds(RANDOM_SEED)

# CIFAR-10 dataset

if __name__ == '__main__':

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.RandomCrop((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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

    class AlexNet(torch.nn.Module):
        def __init__(self, num_classes):
            super().__init__()

            # Feature extractor
            self.features = torch.nn.Sequential(

                # 1st conv layer
                torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=3, stride=2),

                # 2nd conv layer
                torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=3, stride=2),

                # 3rd conv layer
                torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),

                # 4th conv layer
                torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),

                # 5th conv layer
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=3, stride=2)
            )

            self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))

            # Classifier
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(256 * 6 * 6, 4096),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), 256 * 6 * 6)
            logits = self.classifier(x)
            return logits


    model = AlexNet(num_classes=10)

    model = model.to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.1,
                                                           mode='max',
                                                           verbose=True)
    # Load the model if not training
    # model.load_state_dict(torch.load('saved_data/alexnet_cifar10.pt'))
    # optimizer.load_state_dict(torch.load('saved_data/alexnet_cifar10_optimizer.pt'))
    # scheduler.load_state_dict(torch.load('saved_data/alexnet_cifar10_scheduler.pt'))

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

    # Saving
    torch.save(model.state_dict(), 'saved_data/alexnet_cifar10.pt')
    torch.save(optimizer.state_dict(), 'saved_data/alexnet_cifar10_optimizer.pt')
    torch.save(scheduler.state_dict(), 'saved_data/alexnet_cifar10_scheduler.pt')

    model.cpu()
    unnormalizer = UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    class_dict = {0: 'airplane',
                  1: 'automobile',
                  2: 'bird',
                  3: 'cat',
                  4: 'deer',
                  5: 'dog',
                  6: 'frog',
                  7: 'horse',
                  8: 'ship',
                  9: 'truck'}

    show_examples(model=model, data_loader=test_loader, unnormalizer=unnormalizer, class_dict=class_dict)

    mat = compute_confusion_matrix(model=model, data_loader=test_loader, device=torch.device('cpu'))
    plot_confusion_matrix(mat, class_names=class_dict.values())
    plt.show()

