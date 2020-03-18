import torch
import time
import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
from torchsummary import summary

BATCH_SIZE = 8
EPOCHS = 1 
LR = 0.00005
#LR = 0.00001

no_tqdm = True


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # in_channels, out_channels, kernel_size, stride
        self.conv1 = nn.Conv2d(1, 16, 8)
        self.conv2 = nn.Conv2d(16, 64, 8)
        self.conv3 = nn.Conv2d(64, 128, 6, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 4)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 6, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, 8)
        self.deconv4 = nn.ConvTranspose2d(16, 1, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))

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
    summary(net.cuda(), (1, 28, 28))

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

                #inputs = inputs.view(BATCH_SIZE, 784)
                outputs = net(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if(i%100 == 0):
                    pbar.update(100)

                if(i % 100 == 99 and no_tqdm):
                    print("Epoch: {} | iteration: {} | loss avg: {}".format(epoch, i, running_loss/100))
                    running_loss = 0.0
  
            epoch_loss = "| Epoch {}, loss {} |".format(epoch, round(running_loss/total, 8))

    end_time = time.time()
    total_time = str(datetime.timedelta(seconds=round(end_time - start_time, 1)))
    print("\nModel took " + total_time + " (H:M:S).")

    torch.save(net.state_dict(), "data/conv_encoder_net.pth")
