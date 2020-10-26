import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="sin")

# グラフのタイトル
plt.title("sin")

# x軸のラベル
plt.xlabel("x")

# y軸のラベル
plt.ylabel("y")

fig.savefig("img.png")