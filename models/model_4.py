import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import csv
import numpy
import sklearn

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
# print(X)
# print(y)
X = (X - X.mean()) / X.std()

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
# # 转换为PyTorch张量
# X_train = torch.tensor(X_train.values, dtype=torch.float)
# y_train = torch.tensor(y_train.values, dtype=torch.float)
# X_test = torch.tensor(X_test.values, dtype=torch.float)
# y_test = torch.tensor(y_test.values, dtype=torch.float)

# # 对输入张量归一化
# X_train = (X_train - X_train.mean()) / X_train.std()
# X_test = (X_test - X_test.mean()) / X_test.std()

import torch
from torch.nn import Linear
import torch.optim as optim

NSPLITS = 5
mean_r2 = 0
mean_mae = 0
mean_mse = 0
mean_rmse = 0
mean_mape = 0
kf = sklearn.model_selection.KFold(n_splits=NSPLITS, shuffle=True)
for (train_index, test_index) in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # 创建一个线性模型
    model = Linear(in_features=IN_FEATURES, out_features=1)
    torch.nn.init.normal_(model.weight, mean=0.0, std=1.0)
    model.bias.data.fill_(0.)
    # print(model.weight)
    # print(model.bias)
    # 创建一个优化器
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    l1_lambda = 0.0001

    # 创建一个均方误差损失函数
    loss_fn = torch.nn.MSELoss()

    ITERATION = 100000
    for epoch in range(ITERATION):
        if (epoch + 1) % 1000 == 0:  # 每1000次迭代打印一次
                print('Epoch: {:.2f}%'.format((epoch + 1)/ITERATION * 100))
        # 前向传播
        y_pred = model(X_train)

        # 计算损失
        loss = loss_fn(y_pred, y_train)

        # L1正则化
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm  
        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        # 清零梯度
        optimizer.zero_grad()

    print('weight:', model.weight)
    print('bias', model.bias)

    y_test_pred = model(X_test)
    # print((torch.nn.functional.mse_loss(y_test_pred, y_test) / (IN_FEATURES * 0.2)))
    print('y_test_pred:', y_test_pred)
    print('y_test:', y_test)

    # y_test_pred_numpy = y_test_pred.detach().numpy()
    # numpy.savetxt('/Users/linto/Codes/project/output_y_test_pred.csv', y_test_pred_numpy, delimiter=',')

    def measure_accuracy(y_pred, y):
        mean = 0
        for i in range(torch.numel(y)):
            mean += abs(y_pred[i] - y[i])
        return mean / torch.numel(y)

    mean_mse += sklearn.metrics.mean_squared_error(y_test_pred.detach().numpy(), y_test.detach().numpy())
    mean_rmse += numpy.sqrt(sklearn.metrics.mean_squared_error(y_test_pred.detach().numpy(), y_test.detach().numpy()))
    mean_mae += sklearn.metrics.mean_absolute_error(y_test_pred.detach().numpy(), y_test.detach().numpy())
    mean_r2 += sklearn.metrics.r2_score(y_test_pred.detach().numpy(), y_test.detach().numpy())
    mean_mape += sklearn.metrics.mean_absolute_percentage_error(y_test_pred.detach().numpy(), y_test.detach().numpy())
    print('accuracy:', measure_accuracy(y_test_pred, y_test).item())
    print('MSE', sklearn.metrics.mean_squared_error(y_test_pred.detach().numpy(), y_test.detach().numpy()))
    print('RMAE', numpy.sqrt(sklearn.metrics.mean_squared_error(y_test_pred.detach().numpy(), y_test.detach().numpy())))
    print('MAE', sklearn.metrics.mean_absolute_error(y_test_pred.detach().numpy(), y_test.detach().numpy()))
    print('R2', sklearn.metrics.r2_score(y_test_pred.detach().numpy(), y_test.detach().numpy()))
    print('MAPE', sklearn.metrics.mean_absolute_percentage_error(y_test_pred.detach().numpy(), y_test.detach().numpy()))

print('mean_mse:', mean_mse / NSPLITS)
print('mean_rmse:', mean_rmse / NSPLITS)
print('mean_mae:', mean_mae / NSPLITS)
print('mean_r2:', mean_r2 / NSPLITS)
print('mean_mape:', mean_mape / NSPLITS)