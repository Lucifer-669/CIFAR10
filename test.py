import os
import torch
import torch.nn as nn
from src.model import Net
from src.data_loader import load_data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def test_model(model_path='../models/cifar10_net_best.pth'):
    _, testloader, classes = load_data()  # 加载测试数据集
    # 加载训练好的模型
    net = Net().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()  # 设置为评估模式

    # 初始化测试指标
    correct = 0
    total = 0
    class_correct = {cls: 0 for cls in classes}
    class_total = {cls: 0 for cls in classes}

    # 测试循环
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # 总分类准确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 每个类别的准确率
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = classes[labels[i]]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # 计算总体准确率
    overall_accuracy = 100 * correct / total
    print(f'Overall Test Accuracy: {overall_accuracy:.2f}%')

    # 打印每个类别的准确率
    print('\nClass-wise Accuracy:')
    for cls in classes:
        if class_total[cls] > 0:
            acc = 100 * class_correct[cls] / class_total[cls]
            print(f'Accuracy of {cls:5s}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})')
        else:
            print(f'Accuracy of {cls:5s}: No samples in test set.')


if __name__ == '__main__':
    test_model()