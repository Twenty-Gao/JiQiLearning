import numpy as np
import matplotlib.pyplot as plt


x_data =  [1.0, 2.0, 3.0]
y_data =  [2.0, 4.0, 6.0]

def forward(x):
    # 定义的模型
    return x * w

def loss(x, y):
    # 定义损失函数
    # 第一步：求出y的计算值（使用定义的函数）
    y_pred = forward(x)
    # 返回 损失值 （wx-y）*（wx-y）
    return (y_pred - y) * (y_pred - y)


# 权重和权重的损失值 保存到列表中
w_list = []
mse_list = []
# 采样时 的间隔为0.1

for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    #
    for x_val, y_val in zip(x_data, y_data):
        # 计算函数值
        y_pred_val = forward(x_val)
        # 损失值
        loss_val = loss(x_val, y_val)
        # 累加
        l_sum += loss_val
        print("\t", x_val, y_val, y_pred_val, loss_val)
    #在这个打印的时候 除以样本总数，将其转变成Mse
    print("MSE=", l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)


plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()