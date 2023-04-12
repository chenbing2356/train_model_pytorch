import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

class DataProcessing():
    def dataloader(self, train_data_path, test_data_path):
        transform_train=transforms.Compose(
            [
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ]
        )
        transform_valid = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ]
        )

        # 加载数据集-使用pytorch中的公共数据
        trainset = torchvision.datasets.FashionMNIST(root=train_data_path, train=True, download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
        validset = torchvision.datasets.FashionMNIST(root=test_data_path, train=True, download=True, transform=transform_valid)
        validloader = data.DataLoader(validset, batch_size=64, shuffle=True, num_workers=2)
        return trainloader, validloader



