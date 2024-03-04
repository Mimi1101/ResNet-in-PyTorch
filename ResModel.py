import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

#used inchannels and outchannels because n_filter was not working as it threw errors for me.
#I have uploaded my trained model res-net.path along with the output images because the output takes too long to run 
#testing it on 10 epochs because the output takes long to run


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, block):
        identity = block
        out = self.conv1(block)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class ResSect(nn.Module):
    def __init__(self, in_channels, out_channels, n_residual_blocks, beginning_stride):
        super(ResSect, self).__init__()
        self.residual_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=beginning_stride),
            *[ResidualBlock(out_channels, out_channels) for _ in range(n_residual_blocks - 1)]
        )

    def forward(self, x):
        return self.residual_blocks(x)

class ResModel(nn.Module):
    def __init__(self, pretrained=False):
        super(ResModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.sect1 = ResSect(32, 32, 3, 1)
        self.sect2 = ResSect(32, 64, 3, 2)
        self.sect3 = ResSect(64, 128, 3, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 10)

        if pretrained:
            self.load_trained_model()

    def load_trained_model(self):  
        pretrainedModel = 'resnet_cifar10.pth'
        modelLoad = torch.load(pretrainedModel, map_location='cpu')      
        self.load_state_dict(modelLoad)
        print("Pretrained Model Success")
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.sect1(x)
        x = self.sect2(x)
        x = self.sect3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for batchIndex, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batchIndex % 100 == 0:
           print('Epoch: {} Loss: {:.4f}'.format(epoch, loss.item()))


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() 
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
  
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset will also print file already downloaded twice
    traindata = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testdata = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainload = torch.utils.data.DataLoader(traindata, batch_size=64, shuffle=True, num_workers=5)
    testload = torch.utils.data.DataLoader(testdata, batch_size=64, shuffle=False, num_workers=5)
    
    #this is where i am loading the pretrained model
    model = ResModel(pretrained=False)
    model.to(device)
    model.load_trained_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 11):
        train(model, trainload, criterion, optimizer, device)
        test(model, testload, criterion, device)
    print('Finished Training')
    torch.save(model.state_dict(), 'ResNet_cifar10.pth')