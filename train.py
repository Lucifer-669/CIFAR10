import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from src.model import Net
from src.data_loader import load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(num_epochs=50, patience=5, val_ratio=0.1):
    trainloader, _, _ = load_data()

    # 从训练集中划分验证集
    dataset_size = len(trainloader.dataset)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size
    train_set, val_set = random_split(trainloader.dataset, [train_size, val_size])


    trainloader = torch.utils.data.DataLoader(train_set, batch_size=trainloader.batch_size,
                                              shuffle=True, num_workers=2, pin_memory=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=trainloader.batch_size,
                                            shuffle=False, num_workers=2, pin_memory=True)

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    writer = SummaryWriter('../logs/tensorboard')

    best_val_loss = float('inf')  # 初始化最佳验证损失
    epochs_no_improve = 0  # 记录验证损失未改善的 epoch 数

    for epoch in range(num_epochs):
        net.train()
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

            if i % 50 == 49:  # 每50个batch记录一次
                avg_loss = running_loss / 50
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}], Loss: {avg_loss:.3f}')
                writer.add_scalar('Training Loss', avg_loss, epoch * len(trainloader) + i)
                running_loss = 0.0

        # 验证阶段
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(valloader)
        val_accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}] Validation Loss: {avg_val_loss:.3f}, Validation Accuracy: {val_accuracy:.2f}%')
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(net.state_dict(), '../models/cifar10_net_best.pth')  # 保存最佳模型
            print(f'Validation loss improved. Saving best model...')
        else:
            epochs_no_improve += 1
            print(f'Validation loss did not improve. Patience: {epochs_no_improve}/{patience}')
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs!')
                break

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join('../models', f'model_epoch_{epoch + 1}.pth')
            torch.save(net.state_dict(), model_path)
            print(f'Saved model at epoch {epoch + 1} to {model_path}')

        # 更新学习率
        scheduler.step()
        print(f'Epoch [{epoch + 1}] Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

    print('Finished Training')
    writer.close()


if __name__ == '__main__':
    train_model()