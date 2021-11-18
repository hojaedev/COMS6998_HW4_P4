import pickle
from torchvision.models.resnet import (
    resnet50, resnet18
)
import torch
import torchvision
import torchvision.transforms as transforms
from resnet import (
    resnet20, resnet32, resnet44, resnet56
)

depths = [
    (resnet18, 'resnet18'),
    (resnet20, 'resnet20'),
    (resnet32, 'resnet32'),
    (resnet44, 'resnet44'),
    (resnet56, 'resnet56'),
    (resnet50, 'resnet50')
]

def do_test(GPU_TYPE = 'P100'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = 128

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    for model_generator, name in depths:
        model = model_generator()
        model.to(device)
        path_base = f'{name}_{GPU_TYPE}'
        saved = torch.load(path_base + '.model')

        model.load_state_dict(saved['model_state_dict'])
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Name:', name)
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
