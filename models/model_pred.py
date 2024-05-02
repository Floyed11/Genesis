IN_FEATURES = 1740

import torch
from torch import nn, optim
import pandas as pd

# 定义模型
model = nn.Sequential(
    nn.Linear(IN_FEATURES, 64),  # 输入层
    nn.ReLU(),  # 激活函数
    nn.Linear(64, 1)  # 输出层
)
model.load_state_dict(torch.load('model_parameters.pth'))

csv_file_path_X = '/Users/linto/Codes/project/output_X.csv'
data_X = pd.read_csv(csv_file_path_X)
data_X = data_X.apply(pd.to_numeric, errors='coerce')  # 将无法转换为数值的值设置为NaN
data_X = data_X.fillna(0)  # 将NaN值设置为0

X = torch.FloatTensor(data_X.values)

result_y = model(X)
result_y.to_csv('result_y.csv', index=False)
