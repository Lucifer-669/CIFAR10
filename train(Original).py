import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 设置 TensorFlow 日志级别

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.model import Net
from src.data_loader import load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(num_epochs=20):
    trainloader, _, _ = load_data()
    print("Data loaded successfully!")  # 调试信息

    net = Net().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    writer = SummaryWriter('../logs/tensorboard')

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # 每 100 个 batch 打印一次
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                writer.add_scalar('Training Loss', running_loss / 100, epoch * len(trainloader) + i)
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), '../models/cifar10_net.pth')
    writer.close()

if __name__ == '__main__':
    train_model()