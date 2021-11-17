from resnet import (
    resnet20, resnet32, resnet44, resnet56
)
import torch
from torchvision.models.resnet import (
    resnet50, resnet18
)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import pickle
print('------------------------------------')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device', device)
print('------------------------------------')

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

# Training
def train(net, epoch, training_file_path, result, optimizer, criterion, model):
    print('\nEpoch: %d' % (epoch + 1))
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

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        result.append(train_loss/(batch_idx+1))
        
        step = epoch * len(trainloader) + batch_idx + 1

        # with open('/content/drive/My Drive/6998/hw4/'+ file_name, 'a', newline='') as csvfile:
        #     losswriter = csv.writer(csvfile, delimiter=',')
        #     losswriter.writerow((str(step),'%.3f' % (train_loss/(batch_idx+1)) + '\n'))
        """
        with open('/content/drive/My Drive/6998/hw4/'+ file_name, 'a') as testwritefile:
          testwritefile.write(str(step) + ' ' + '%.3f' % (train_loss/(batch_idx+1)) + '\n')
        """
        
    save_checkpoint(model, optimizer, training_file_path, epoch)


    
    
def do_train(GPU_TYPE = 'P100'):
    depths = [
        (resnet18, 'resnet18'),
        (resnet20, 'resnet20'),
        (resnet32, 'resnet32'),
        (resnet44, 'resnet44'),
        (resnet56, 'resnet56'),
        (resnet50, 'resnet50')
    ]

    EPOCHS = 150

    for net, name in depths:
        # Model
        model = net()
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        training_file_path_base = f'{name}_{GPU_TYPE}'

        result = []
        for epoch in range(EPOCHS):
            train(model, epoch, training_file_path_base + '.model', result, optimizer, criterion, model)
            scheduler.step()

        with open(training_file_path_base + '.pickle', 'wb') as f:
            pickle.dump(result, f)
