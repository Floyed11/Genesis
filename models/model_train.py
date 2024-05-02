import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import csv
import numpy
import sklearn
import math

IN_FEATURES = 1740

'''输入数据集'''
csv_file_path_X = '/Users/linto/Codes/project/output_X.csv'
data_X = pd.read_csv(csv_file_path_X)
# data_X = data_X.astype()

data_X = data_X.apply(pd.to_numeric, errors='coerce')  # 将无法转换为数值的值设置为NaN
data_X = data_X.fillna(0)  # 将NaN值设置为0

csv_file_path_y = '/Users/linto/Codes/project/output_y.csv'
data_y = pd.read_csv(csv_file_path_y)

data_y = data_y.apply(pd.to_numeric, errors='coerce')  # 将无法转换为数值的值设置为NaN
data_y = data_y.fillna(0)  # 将NaN值设置为0

'''查看输出'''
X = torch.FloatTensor(data_X.values)
y = torch.FloatTensor(data_y.values)

X = (X - X.mean()) / X.std()

import torch
from torch import nn, optim

# 定义模型
model = nn.Sequential(
    nn.Linear(IN_FEATURES, 64),  # 输入层
    nn.ReLU(),  # 激活函数
    nn.Linear(64, 1)  # 输出层
)

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 训练
ITERATION = 500000
for epoch in range(ITERATION):
    if (epoch + 1) % 1000 == 0:  # 每1000次迭代打印一次
            print('Epoch: {:.2f}%'.format((epoch + 1)/ITERATION * 100))

    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播
    loss.backward()

    # 更新权重
    optimizer.step()

    # 清零梯度
    optimizer.zero_grad()

    
# 保存模型
torch.save(model.state_dict(), 'model_parameters.pth')