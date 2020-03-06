import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200,10)
        self.fc4 = nn.Linear(10, 2)
        
        self.fc5 = nn.Linear(2, 10)
        self.fc6 = nn.Linear(10, 200)
        self.fc7 = nn.Linear(200, 500)
        self.fc8 = nn.Linear(500, 784)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.sigmoid(self.fc8(x))

        return x

# returns trainloader and testloader
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        ])

    trainset = datasets.MNIST('data', train=True, download=True,
            transform=transform)

    testset = datasets.MNIST('data', train=False, download=True,
            transform=transform)


    trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=10, shuffle=True)

    testloader = torch.utils.data.DataLoader(testset,
            batch_size=10, shuffle=True)
    
    #dataiter = iter(testloader)
    #images, labels = dataiter.next()
    #print(images)

    return(trainloader, testloader)
    
net = Net()
print(net)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

trainloader, testloader = load_data()

for epoch in range(2):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        inputs = inputs.view(10,784)
        outputs = net(inputs)
        loss = criterion(outputs, inputs) # 784, 10... 
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print("[{},{} loss: {}".format(epoch+1, i+1, running_loss/2000))
            running_loss = 0.0


torch.save(net.state_dict(), 'data/encoder_net.pth')

