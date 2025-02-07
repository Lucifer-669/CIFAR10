import torch
from src.model import Net
from src.data_loader import load_data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def test_model():
    _, testloader, classes = load_data()
    net = Net().to(device)
    net.load_state_dict(torch.load('../models/cifar10_net.pth'))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

if __name__ == '__main__':
    test_model()