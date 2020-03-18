import torch
import time
import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms

BATCH_SIZE = 16
EPOCHS = 10
LR = 0.00005

no_tqdm = False


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 10)
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
        x = torch.sigmoid(self.fc8(x))

        return x


# returns trainloader and testloader
def load_data():
    transform = transforms.Compose([transforms.ToTensor(), ])

    trainset = datasets.MNIST("data", train=True, download=True, transform=transform)

    testset = datasets.MNIST("data", train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=True
    )

    return (trainloader, testloader)


trainloader, testloader = load_data()

if __name__ == "__main__":
    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net()
    print(net)

    net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR, )

    total = 60000/BATCH_SIZE
    
    epoch_loss = ''


    for epoch in range(EPOCHS):
        running_loss = 0.0

        with tqdm(total=total, unit='batches', postfix=epoch_loss, disable=no_tqdm) as pbar:
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                inputs = inputs.view(BATCH_SIZE, 784)
                outputs = net(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if(i%100 == 0):
                    pbar.update(100)

                if(i % 500 == 499 and no_tqdm):
                    print("Epoch: {} | iteration: {} | loss avg: {}".format(epoch, i, running_loss/500))
                    running_loss = 0.0
  
            epoch_loss = "| Epoch {}, loss {} |".format(epoch, round(running_loss/total, 8))

    end_time = time.time()
    total_time = str(datetime.timedelta(seconds=round(end_time - start_time, 1)))
    print("\nModel took " + total_time + " (H:M:S).")

    torch.save(net.state_dict(), "data/encoder_net.pth")
