import numpy as np
import matplotlib.pyplot as plt
# 梯度下降

x_data =  [1.0, 2.0, 3.0]
y_data =  [2.0, 4.0, 6.0]

# 超参数的初始化
w = 1.0
def forward(x):
    # 前馈计算
    return x * w


# 损失函数
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) * (y_pred - y)
    return cost / len(xs)


# 梯度函数
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

print("Predict (before training)", 4, forward(4))
#  训练过程
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w = w - 0.01 * grad_val
    print("Epoch:", epoch, "w=", w, "cost=", cost_val)
print("Predict (after training)", 4, forward(4))