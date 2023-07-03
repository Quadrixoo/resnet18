import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import os




#Hyper-Parameter
num_epochs = 1000
batch_size = 128
learning_rate = 0.001
lable_noise = 0.15
width_param = 20
device = 'cuda'         #if installed, otherwise use 'cpu' (not recommended)
seed = 0

severity_lvl = 5                    #look in the documentation for CIFAR-10-C
corruption_param = "pixelate"



#Setting the Seed for reproducibility
torch.manual_seed(seed)


# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = os.environ.get("PATH_DATASETS", "../data/")

train_dataset = CIFAR10(root='./data', train=True, download=False)
DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
DATA_STD = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, width):
        super(ResNet, self).__init__()
        self.in_planes = int(64*width/10)

        self.conv1 = nn.Conv2d(3, int(64*width/10), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64*width/10))
        self.layer1 = self._make_layer(block, int(64*width/10), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*width/10), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*width/10), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*width/10), num_blocks[3], stride=2)
        self.linear = nn.Linear(int(512*width/10)*block.expansion, 10)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(width):
    return ResNet(BasicBlock, [2,2,2,2], width)


#implementing the dataloader for CIFAR-10-C
def imshow(loader):                                 #shows corrupted images 
    dataiter = iter(loader)
    images, labels = next(dataiter)
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_loader(corruption, batch_size=128, severity=5):
    xs = transforms.Normalize(DATA_MEANS, DATA_STD)(torch.from_numpy(np.load('./data/CIFAR-10-C/{}.npy'.format(corruption)) / 255.).float().transpose(1, 2).transpose(1, 3))
    ys = torch.from_numpy(np.load('./data/CIFAR-10-C/labels.npy'))
    # only pick the last 10000 as these have the highest corruption level
    #n_data = xs.shape[0]
    xs = xs[(10000*(severity-1)):(10000*severity)]
    ys = ys[(10000*(severity-1)):(10000*severity)]

    ood_set = torch.utils.data.TensorDataset(xs, ys)
    # We define a set of data loaders that we can use for various purposes later.
    return torch.utils.data.DataLoader(ood_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)

corruptions = [
    'None',
    'gaussian_blur',
    'brightness',
    'speckle_noise',
    'glass_blur',
    'spatter',
    'shot_noise',
    'defocus_blur',
    'elastic_transform',
    'frost',
    'saturate',
    'snow',
    'gaussian_noise',
    'contrast',
    'motion_blur',
    'impulse_noise',
    'pixelate',
    'fog',
    'jpeg_compression',
    'zoom_blur',
]

#Loading checkpoints and testing the ResNet on the CIFAR10-C dataset 
def test_resnet18(width, epochs=10, batch_size=128, lr=0.001, noise_level=0.15, device='cuda', corruption='speckle_noise', severity=5):
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    

    #train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    #test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    loader = get_loader(corruption, batch_size, severity)

    net = ResNet18(width).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    #setting list with epochs of checkpoints
    lst = list(range(0,30))
    lst2 = []
    num = 150
    for i in range(18):
        lst2 = lst2 + [num]
        num += 50
    epochs = lst + [40,60,80,100] + lst2

    #testing the ResNet
    for epoch in epochs:
        
        #Loading Checkpoint
        net.load_state_dict(torch.load("./resnet" + str(width) + "_" + str(epoch) +".ckpt", map_location=device))

        # Test the model
        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data in loader:
                dataiter = iter(loader)
                images, labels = next(dataiter)
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Epoch %d: Accuracy of the network on the 10000 test images: %d %%' % (
                    epoch + 1, 100 * correct / total))
                  
    
    #If the results are supposed to be safed in a seperate .txt document

        # datei = open("ResNet18_"+str(width_param)+str(corruption)+str(severity)+".txt", "a+")         #speichert Accuracy in textdatei
        # datei.write('Epoch %d: Accuracy of the network on the CIFAR10-C images: %d %%\n' % (
        #             epoch + 1, 100 * correct / total))
        
        
        
test_resnet18(width_param, epochs=10, batch_size=100, lr=0.001, noise_level=0.15, device='cuda', corruption=corruption_param, severity=severity_lvl)      