import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
    


def load_data():
    train_img = np.load('data/train_img.npy')
    train_labels = np.load('data/train_labels.npy')

    test_img = np.load('data/test_img.npy')
    test_labels = np.load('data/test_labels.npy')

    s = np.arange(train_img.shape[0])
    np.random.shuffle(s)

    train_img = train_img[s]
    train_labels = train_labels[s]
    

    x_train = torch.from_numpy(train_img[:60000]/255)
    indicies = torch.from_numpy(train_labels[:60000].reshape(1,-1)).long()-1
    y_train = F.one_hot(indicies, 10)


    x_val = torch.from_numpy(train_img[60000:]/255)
    indicies = torch.from_numpy(train_labels[60000:].reshape(1,-1)).long()-1
    y_val = F.one_hot(indicies, 10)

    s = np.arange(test_img.shape[0])
    np.random.shuffle(s)

    x_test = torch.from_numpy(test_img[s]/255)
    indicies = torch.from_numpy(test_labels[s].reshape(1,-1)).long()-1
    y_test = F.one_hot(indicies, 10)

    return x_train, y_train, x_val, y_val, x_test, y_test


net = Net()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001)

x_train, y_train, x_val, y_val, x_test, y_test = load_data()


for epoch in range(2):
    running_loss = 0.0

    for i in range(6, len(x_train), 6):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backwards + optimize
        outputs = net(x_train[i-6:i])
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # print stats
        if i % 2000 == 1999:
            print("[{},{} loss: {}".format(epoch+1, i+1, running_loss/2000))
            running_loss = 0


torch.save(net.state_dict(), './net.pth')
