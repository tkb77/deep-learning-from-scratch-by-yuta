import numpy as np
import matplotlib.pyplot as plt

def step_func(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

fig = plt.figure()

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_func(x)
y3 = relu(x)

plt.plot(x, y1, label="sigmoid")
plt.plot(x, y2, linestyle="--", label="step_func")
plt.plot(x, y3, linestyle="dashdot", label="relu")

# グラフのタイトル
plt.title("activation function")

# x軸のラベル
plt.xlabel("x")

# y軸のラベル
plt.ylabel("y")

fig.savefig("img.png")