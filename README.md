# CIFAR-10 图像分类项目

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-green)](https://github.com/yourusername/cifar10-classifier)

## 项目简介
本项目旨在使用 PyTorch 框架开发一个卷积神经网络（CNN）模型，对 CIFAR-10 数据集进行图像分类。通过数据预处理、优化策略和模型设计，最终在测试集上达到了 **80.18%** 的准确率。（使用官方的教程所训练出的模型的平均准确率仅为50%左右，见Original代码）
![f0eb03675786a7aae4a7f46fcba7a618](https://github.com/user-attachments/assets/39c15727-d2eb-4935-8a9c-6fb8d019e584)

## 技术栈
- **框架**：PyTorch
- **技术**：卷积神经网络（CNN）、数据增强、早停机制、学习率调度、Dropout、Batch Normalization

## 模型架构
模型包含以下主要组件：
1. **卷积层**：
   - 第一层：`nn.Conv2d(3, 32, 3, padding=1)`，将输入通道数从 3 增加到 32。
   - 第二层：`nn.Conv2d(32, 64, 3, padding=1)`，将通道数从 32 增加到 64。
2. **池化层**：
   - `nn.MaxPool2d(2, 2)`，每次将特征图尺寸减半。
3. **全连接层**：
   - 第一层：`nn.Linear(64 * 8 * 8, 512)`，输入维度为 `64 * 8 * 8`，输出维度为 512。
   - 第二层：`nn.Linear(512, 128)`，输入维度为 512，输出维度为 128。
   - 第三层：`nn.Linear(128, 10)`，输出 10 个类别的 logits。
4. **Dropout**：
   - 在全连接层之间使用 `nn.Dropout(0.25)`，防止过拟合。

## 数据预处理与增强
- **归一化**：使用 CIFAR-10 数据集的均值和标准差进行归一化。
- **数据增强**：使用随机裁剪和水平翻转，提高模型的泛化能力。

## 训练与验证
- **优化器**：Adam 优化器，学习率 0.001，权重衰减 1e-4。
- **学习率调度**：使用 `StepLR` 调度器，每 15 个 epoch 学习率乘以 0.1。
- **早停机制**：如果验证损失在连续 5 个 epoch 内没有改善，提前停止训练。
- **验证集性能**：在验证集上达到了 **77.66%** 的准确率。

## 测试结果
- **测试集准确率**：在测试集上达到了 **80.18%** 的准确率。
- ![7f354f7f81f9fbff2ecbc7fdd870a0d9](https://github.com/user-attachments/assets/dad8c92c-6f77-4ef4-8363-738ee13aa5da)

## 项目成果
- **高效模型**：通过优化模型结构和训练策略，显著提高了模型的训练效率和泛化能力。
- **最佳模型保存**：在验证集上性能最好的模型被保存下来，确保最终模型具有最佳性能。

## 项目结构
![image](https://github.com/user-attachments/assets/e0007dc3-a735-4464-94d4-b305c10e6124)
