import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import random
import os



#Hyper-Parameter
num_epochs = 1001
batch_size = 100
learning_rate = 0.001
lable_noise = 0.15
width_param = 20
device = 'cuda'                 #if installed, else 'cpu' (not recommended)
seed = 0

#Setting the Seed for reproducibility
torch.manual_seed(seed)

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



#This is where the training happens
def train_resnet18(width, epochs=10, batch_size=128, lr=0.001, noise_level=0.15, device='cuda'):
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Add noise to training labels
    for i in range(len(train_set)):
        if random.random() < noise_level:
            train_set.targets[i] = random.randint(0, 9)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    net = ResNet18(width).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))


            #If the results are supposed to be safed in seperate .txt file
                # datei = open("ResNet18_"+str(width_param)+"_"+str(seed)+".txt", "a+")         #speichert loss in textdatei
                # datei.write('[%d, %5d] loss: %.3f\n' %
                #       (epoch + 1, i + 1, running_loss / 100))
                
                _, predicted = torch.max(outputs.data, 1)      #werted accuracy auf training set aus
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            

            #If the results are supposed to be safed in seperate .txt file
                # datei = open("ResNet18_"+str(width_param)+"_"+str(seed)+".txt", "a+")             #speichert accuracy in textdatei
                # datei.write('Epoch %d: Accuracy of the network on the train set: %d %%\n' % (
                #         epoch + 1, 100 * correct / total))
            
                running_loss = 0.0

        #Safing the Accuracy and checkpoints in growing intervals
        if epoch <= 30:
            
            #saving the checkpoint
            torch.save(net.state_dict(), 'resnet'+str(width_param)+'_'+str(seed)+'_'+str(epoch)+'.ckpt')
            
            #testing the resnet on the test set
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Epoch %d: Accuracy of the network on the 10000 test images: %d %%' % (
                    epoch + 1, 100 * correct / total))

        #If the results are supposed to be safed in seperate .txt file
            # datei = open("ResNet18_"+str(width_param)+"_"+str(seed)+".txt", "a+")             #speichert accuracy in textdatei
            # datei.write('Epoch %d: Accuracy of the network on the 10000 test images: %d %%\n' % (
            #         epoch + 1, 100 * correct / total))
            
            
        elif epoch <= 100:

            if epoch % 20 == 0:
                #saving checkpoint
                torch.save(net.state_dict(), 'resnet'+str(width_param)+'_'+str(seed)+'_'+str(epoch)+'.ckpt')
                
                #testing the resnet on the test set
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in test_loader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print('Epoch %d: Accuracy of the network on the 10000 test images: %d %%' % (
                        epoch + 1, 100 * correct / total))

            #If the results are supposed to be safed in seperate .txt file
                # datei = open("ResNet18_"+str(width_param)+"_"+str(seed)+".txt", "a+")             #speichert accuracy in textdatei
                # datei.write('Epoch %d: Accuracy of the network on the 10000 test images: %d %%\n' % (
                #         epoch + 1, 100 * correct / total))
                
                
        else:
            
            if epoch % 50 == 0:
                #saving checkpoint
                torch.save(net.state_dict(), 'resnet'+str(width_param)+'_'+str(seed)+'_'+str(epoch)+'.ckpt')
                
                #testing the resnet on the test set
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in test_loader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print('Epoch %d: Accuracy of the network on the 10000 test images: %d %%' % (
                        epoch + 1, 100 * correct / total))

            #If the results are supposed to be safed in seperate .txt file
                # datei = open("ResNet18_"+str(width_param)+"_"+str(seed)+".txt", "a+")             #speichert accuracy in textdatei
                # datei.write('Epoch %d: Accuracy of the network on the 10000 test images: %d %%\n' % (
                #         epoch + 1, 100 * correct / total))
                
            
                
                
            

            
        
    print('Finished Training')


train_resnet18(width_param, num_epochs, batch_size, learning_rate, lable_noise, device)