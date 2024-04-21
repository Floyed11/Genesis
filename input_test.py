import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

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

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
# 转换为PyTorch张量
X_train = torch.tensor(X_train.values, dtype=torch.float)
y_train = torch.tensor(y_train.values, dtype=torch.float)
X_test = torch.tensor(X_test.values, dtype=torch.float)
y_test = torch.tensor(y_test.values, dtype=torch.float)

# 对输入张量归一化
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()
'''查看输出'''
#print(X_train)
# print(X_train.shape)
#print(y_train)
# print(y_train.shape)
# print(X_test)
# print(X_test.shape)
# print(y_test)
# print(y_test.shape)

# # 创建数据加载器
# train_dataset = TensorDataset(X_train, y_train)
# train_loader = DataLoader(train_dataset, batch_size=32)

# test_dataset = TensorDataset(X_test, y_test)
# test_loader = DataLoader(test_dataset, batch_size=32)