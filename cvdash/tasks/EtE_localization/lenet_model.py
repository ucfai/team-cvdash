import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 4)

        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    

# https://pytorch.org/docs/stable/_modules/torchvision/datasets/svhn.html
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ])
    trainset = datasets.SVHN('data', split='train', download=True, transform=transform)
    #trainset.labels = np.eye(10)[np.array([trainset.labels]).reshape(-1)]
    #print(trainset[0])

    testset = datasets.SVHN('data', split='test', download=True, transform=transform)
    #testset.labels = np.eye(10)[np.array([testset.labels]).reshape(-1)]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)

    
    return(trainloader, testloader)

net = Net()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01)

trainloader, testloader = load_data()

for epoch in range(2):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        if(i==0):
            print(labels[0])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backwards + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print stats
        if i % 2000 == 1999:
            print(inputs[0])
            print("[{},{} loss: {}".format(epoch+1, i+1, running_loss/2000))
            running_loss = 0.0


torch.save(net.state_dict(), './net.pth')
