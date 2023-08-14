import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from helper_data import get_dataloaders_mnist
from helper_train import train_autoencoder_v1
from helper_utils import set_deterministic, set_all_seeds
from helper_plotting import plot_training_loss, plot_generated_images, plot_latent_space_with_labels


def main():

    # Settings
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: ' + str(device))

    RANDOM_SEED = 123
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32
    NUM_EPOCHS = 20

    # set_deterministic()
    set_all_seeds(RANDOM_SEED)

    # MNIST dataset
    train_loader, valid_loader, test_loader = get_dataloaders_mnist(
        batch_size=BATCH_SIZE,
        num_workers=2,
        validation_fraction=0.)

    print('Training Set:\n')
    for images, labels in train_loader:
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break

    # Model
    class Reshape(nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.shape = args

        def forward(self, x):
            return x.view(self.shape)

    class Trim(nn.Module):
        def __init__(self, *args):
            super().__init__()

        def forward(self, x):
            return x[:, :, :28, :28]

    class AutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = nn.Sequential( # 784 MNIST input -> 2 latent space
                nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.Flatten(),
                nn.Linear(3136, 2)
            )
            self.decoder = nn.Sequential( # 2 latent space -> 784 MNIST output
                torch.nn.Linear(2, 3136),
                Reshape(-1, 64, 7, 7),
                nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0),
                Trim(), # 1x29x29 -> 1x28x28
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    set_all_seeds(RANDOM_SEED)
    model = AutoEncoder()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training
    log_dictionary = train_autoencoder_v1(num_epochs=NUM_EPOCHS,
                                    model=model,
                                    optimizer=optimizer,
                                    device=device,
                                    train_loader=train_loader,
                                    skip_epoch_stats=True,
                                    logging_interval=250)

    plot_training_loss(log_dictionary['train_loss_per_batch'], NUM_EPOCHS)
    plt.show()

    plot_generated_images(data_loader=train_loader, model=model, device=device)
    plot_latent_space_with_labels(
        num_classes=10,
        data_loader=train_loader,
        model=model,
        device=device
    )

    plt.legend()
    plt.show()

    with torch.no_grad():
        new_image = model.decoder(torch.tensor([2.5, -2.5]).to(device))
        new_image.squeeze_(0)
        new_image.squeeze_(0)
    plt.imshow(new_image.to('cpu').numpy(), cmap='binary')
    plt.show()


if __name__ == '__main__':
    main()


