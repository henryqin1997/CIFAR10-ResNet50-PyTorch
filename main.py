'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import datetime

from models import *
from utils import progress_bar
import json


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1.0, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='WE',
                    help='number of warmup epochs (default: 5)')
parser.add_argument('--lr-decay', nargs='+', type=int, default=[50, 75],
                    help='epoch intervals to decay lr')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                    help='SGD weight decay (default: 5e-4)')
parser.add_argument('--optimizer',type=str,default='sgd',
                    help='different optimizers')
parser.add_argument('--max-lr',default=0.1,type=float)
parser.add_argument('--div-factor',default=25,type=float)
parser.add_argument('--final-div',default=10000,type=float)
parser.add_argument('--num-epoch',default=150,type=int)
parser.add_argument('--pct-start',default=0.3,type=float)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='/tmp/cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='/tmp/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = ResNet50()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()


if args.optimizer.lower()=='sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.optimizer.lower()=='sgdwm':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)
elif args.optimizer.lower()=='adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
elif args.optimizer.lower() == 'rmsprop':
    optimizer = optim.RMSprop(net.parameters(),lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)
elif args.optimizer.lower() == 'adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer.lower() == 'radam':
    from radam import RAdam
    optimizer = RAdam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
elif args.optimizer.lower() == 'lars':#no tensorboardX
    from lars import Lars
    optimizer = Lars(net.parameters(), lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
elif args.optimizer.lower() == 'lamb':
    from lamb import Lamb
    optimizer  = Lamb(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
elif args.optimizer.lower() == 'novograd':
    from novograd import NovoGrad
    optimizer = NovoGrad(net.parameters(), lr=args.lr,weight_decay=args.weight_decay)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)



lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,args.max_lr,steps_per_epoch=len(trainloader),
                                                   epochs=args.num_epoch,div_factor=args.div_factor,final_div_factor=args.final_div,pct_start=args.pct_start)


train_acc = []
valid_acc = []


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_acc.append(correct/total)

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    valid_acc.append(correct/total)

for epoch in range(args.num_epoch):
    train(epoch)
    test(epoch)

print("Best test accuracy: {}".format(max(valid_acc)))

fn = '{}{}-{}-epoch{}-batchsize{}-pct{}-{}_onecycle_log.json'.format(args.optimizer,str(args.max_lr/args.div_factor),
                            str(args.max_lr),args.num_epoch,args.batch_size,args.pct_start,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

file = open(fn,'w+')
json.dump([train_acc,valid_acc],file)