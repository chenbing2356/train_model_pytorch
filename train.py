import torch
import torch.nn as nn
from data_processing import DataProcessing
import torch.optim as optim
from network import Net
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as data
from tqdm import tqdm

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(Net().parameters(), lr=0.001, momentum=0.9)

# def Trainer(epoch_num):
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
trainset = torchvision.datasets.FashionMNIST(root=r"./data/train", train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=16, shuffle=True)
validset = torchvision.datasets.FashionMNIST(root=r"./data/test", train=True, download=True, transform=transform_valid)
validloader = data.DataLoader(validset, batch_size=1, shuffle=True)


# 训练
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader), 0):

        inputs, label = data
        network = Net()
        outputs = network(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        # 梯度清零

    print("epoch[{}/{}],average loss:{:.4f}".format(epoch+1, 100, running_loss/len(trainloader)))
