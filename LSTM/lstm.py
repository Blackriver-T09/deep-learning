import os
import torch   #torch用于深度学习
# torch 是 PyTorch 中用于处理张量（多维数组）的库.类似于 NumPy，但支持 GPU 加速，且提供自动微分功能
import numpy as np
import pandas as pd

from torch import nn, optim   #torch中的子模块，用于定义神经网络和优化器。  
from torch.utils.data import DataLoader  #torch中的数据加载器，用于批量加载数据。
from torchvision import datasets, transforms  #torchvision中的模块，用于处理和转换图像数据。


batchsize=100   #设置每次训练和测试批量大小为100。
# 从本地路径加载Fashion MNIST数据集，并将图像数据转换为Tensor。
# Tensor（张量） 是一种数据格式，可以看作是多维数组或矩阵的推广。Tensor 的优势在于它可以在 CPU 和 GPU 上高效地执行数值运算
# Fashion MNIST 是一个包含10类时尚商品图片的数据集。
training_data = datasets.FashionMNIST(root=r"C:\Users\28121\Desktop\AI_data\Fashion_MNIST", train=True, transform=transforms.ToTensor(), download=True)  
test_data = datasets.FashionMNIST(root=r"C:\Users\28121\Desktop\AI_data\Fashion_MNIST", train=False, transform=transforms.ToTensor(), download=True)



# 使用DataLoader创建数据加载器，用于批量读取训练和测试数据。
train_dataloader = DataLoader(training_data, batch_size=batchsize)
test_dataloader = DataLoader(test_data, batch_size=batchsize)



# 超参数hyperparmaters
# 超参数（Hyperparameters） 是在训练模型之前设置的参数，它们不会在训练过程中更新，而是用于控制模型训练的过程和结构。
sequence_len = 28
input_len = 28       #LSTM的输入形状，表示图像每行的像素作为一个时间步。 时间步（Time Step） 是在时间序列数据或序列模型（如RNN, LSTM）中，表示序列中每个元素所处的位置。
hidden_size = 128    #LSTM的隐藏层大小。隐藏层（Hidden Layer） 是神经网络中位于输入层和输出层之间的层，用于提取和学习数据的特征
num_layers = 2       #LSTM的层数
num_classes = 10     #分类的类别数，Fashion MNIST数据集有10个类别。
num_epochs = 5       #训练的迭代次数。
learning_rate = 0.01 #学习率，控制优化器的步长。



# 定义LSTM模型
class LSTM(nn.Module): 
    def __init__(self, input_len, hidden_size, num_class, num_layers):#初始化LSTM模型的结构，包括LSTM层和输出层（全连接层）
        super(LSTM, self).__init__()  
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_len, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_classes)
    
    def forward(self, X):  #forward方法定义了前向传播的计算过程
        hidden_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)  #创建初始化的隐藏状态和细胞状态。
        cell_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)   
        out, _ = self.lstm(X, (hidden_states, cell_states))
        out = self.output_layer(out[:, -1, :])
        return out
    

# 实例化LSTM模型，并打印模型结构。
model = LSTM(input_len, hidden_size, num_classes, num_layers)
print(model)

# 使用交叉熵损失函数，适用于多分类问题。
# 交叉熵损失函数（Cross-Entropy Loss） 是一个常用于分类任务的损失函数。
# 它衡量的是模型预测的概率分布与真实标签的概率分布之间的差异。对于多分类问题，交叉熵损失特别有效。
loss_func = nn.CrossEntropyLoss()


# 定义了两个优化器SGD和Adam，用于调整模型参数，后续训练时会使用Adam优化器。
# 优化器（Optimizer） 是用于更新和计算网络参数以最小化（或最大化）一个目标函数（通常是损失函数）的算法。
# SGD（随机梯度下降）：最基本的优化器，每次更新只考虑一个样本或一小批样本的梯度。
# Momentum：在SGD的基础上添加了动量项，帮助优化器在正确的方向上加速，避免了过多的震荡。
# Adam（Adaptive Moment Estimation）：结合了动量和自适应学习率技术的优化器，能够根据每个参数的性质自动调整学习率。
sgd = optim.SGD(model.parameters(), lr=learning_rate)
adam = optim.Adam(model.parameters(), lr=learning_rate)




# 定义了模型的训练过程
def train(num_epochs, model, train_dataloader, loss_func, optimizer):
    total_steps = len(train_dataloader)

    for epoch in range(num_epochs):  #迭代所有的训练数据，进行多个epoch训练。
        for batch, (images, labels) in enumerate(train_dataloader):
            images = images.reshape(-1, sequence_len, input_len)  #每个batch的图像被调整为LSTM模型的输入形状（sequence_len × input_len）。
            
            output = model(images)  #前向传播计算输出，计算损失，反向传播更新模型参数。
            loss = loss_func(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch+1)%100 == 0:   #每100个batch打印一次损失值。
                print(f"Epoch: {epoch+1}; Batch {batch+1} / {total_steps}; Loss: {loss.item():>4f}")



train(num_epochs, model, train_dataloader, loss_func, adam)  #调用train函数，用Adam优化器训练模型。

# 随机从测试数据集中取出一个batch的图像和标签。
test_images, test_labels = next(iter(test_dataloader))
print(test_labels)

# 调整图像形状并进行前向传播，得到预测结果。
test_output = model(test_images.view(-1, 28, 28))
predicted = torch.max(test_output, 1)[1]
print(predicted)

# 计算预测正确的样本数，并计算准确率。
correct = [1 for i in range(100) if predicted[i] == test_labels[i]]
percentage_correct = sum(correct)/100
print(percentage_correct)



# test_loop函数定义了模型的测试过程
def test_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad(): #不计算梯度（torch.no_grad()），仅进行前向传播。
        for X, y in dataloader:  #迭代所有测试数据，计算总的损失和准确率。
            # reshape images
            X = X.reshape(-1, 28, 28)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error:\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    return 100*correct


test_loop(test_dataloader, model, loss_func, adam)  #调用test_loop函数