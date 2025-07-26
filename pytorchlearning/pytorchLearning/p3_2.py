import numpy as np
import matplotlib.pyplot as plt
# 随机梯度下降

x_data =  [1.0, 2.0, 3.0]
y_data =  [2.0, 4.0, 6.0]

# 超参数的初始化
w = 1.0
def forward(x):
    # 前馈计算
    return x * w


# 损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2

# 梯度函数
def gradient(x, y):
    return 2 * x * ( x * w - y)

print("Predict (before training)", 4, forward(4))
#  训练过程
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01 * grad
        print("\tgrad:", x, y,grad)
        l = loss(x, y)
    print("progress:", epoch, "w=", w, "loss=", l)
print("Predict (after training)", 4, forward(4))