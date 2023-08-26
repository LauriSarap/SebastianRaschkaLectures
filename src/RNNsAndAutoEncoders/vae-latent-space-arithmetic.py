import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from helper_data import get_dataloaders_celeba
from helper_data import compute_average_faces
from helper_plotting import plot_modified_faces


def main():

    # Settings
    CUDA_DEVICE_NUM = 0
    DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
    print(f'Using {DEVICE} device')

    RANDOM_SEED = 123
    BATCH_SIZE = 5000

    # Data
    custom_transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((128, 128)),
        torchvision.transforms.ToTensor()
    ])

    train_loader, valid_loader, test_loader = get_dataloaders_celeba(
        batch_size=BATCH_SIZE,
        train_transforms=custom_transforms,
        test_transforms=custom_transforms,
        num_workers=2
    )

    torch.manual_seed(RANDOM_SEED)
    for images, labels in train_loader:
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break

    EXAMPLE_IMAGE = images[1]

    # Image manipulation in original space


if __name__ == '__main__':
    main()
