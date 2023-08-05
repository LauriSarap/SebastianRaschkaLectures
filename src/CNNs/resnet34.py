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

    # Cifar-10 dataset
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((120, 120)),
        torchvision.transforms.RandomCrop((110, 110)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((120, 120)),
        torchvision.transforms.CenterCrop((110, 110)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=BATCH_SIZE,
        validation_fraction=0.1,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        num_workers=2)

    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        print('Class labels of 10 examples:', labels[:10])
        break

    # Model
    def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
        return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)

    def conv1x1(in_planes, out_planes, stride=1):
        return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    class BasicBlock(torch.nn.Module):
        expansion: int = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None,
                     groups=1, base_width=64, dilation=1, norm_layer=None):

            super().__init__()
            if norm_layer is None:
                norm_layer = torch.nn.BatchNorm2d
            if groups != 1 or base_width != 64:
                raise ValueError('BasicBlock only supports groups=1 and base_width=64')
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = norm_layer(planes)
            self.relu = torch.nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    class Bottleneck(torch.nn.Module):

        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None,
                     groups=1, base_width=64, dilation=1, norm_layer=None):

            super().__init__()
            if norm_layer is None:
                norm_layer = torch.nn.BatchNorm2d
            width = int(planes * (base_width / 64.)) * groups
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = norm_layer(width)
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
            self.relu = torch.nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    class ResNet(torch.nn.Module):

        def __init__(self, block, layers, num_classes, zero_init_residual=False, groups=1,
                     width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):

            super().__init__()
            if norm_layer is None:
                norm_layer = torch.nn.BatchNorm2d
            self._norm_layer = norm_layer

            self.inplanes = 64
            self.dilation = 1
            if replace_stride_with_dilation is None:
                replace_stride_with_dilation = [False, False, False]
            if len(replace_stride_with_dilation) != 3:
                raise ValueError("replace_stride_with_dilation should be None "
                                 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            self.groups = groups
            self.base_width = width_per_group
            self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                         bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = torch.nn.ReLU(inplace=True)
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)

            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        torch.nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        torch.nn.init.constant_(m.bn2.weight, 0)

        def _make_layer(self, block, planes, blocks,
                        stride=1, dilate=False):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = torch.nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

            return torch.nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10) # ResNet-34

    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, verbose=True)

    minibatch_loss_list, train_acc_list, valid_acc_list = train_model(
        model = model,
        num_epochs=NUM_EPOCHS,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=DEVICE,
        scheduler=scheduler,
        scheduler_on='valid_acc',
        logging_interval=100
    )

    torch.save(model.state_dict(), 'saved_data/resnet34.pt')
    torch.save(optimizer.state_dict(), 'saved_data/resnet34_optimizer.pt')
    torch.save(scheduler.state_dict(), 'saved_data/resnet34_scheduler.pt')

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