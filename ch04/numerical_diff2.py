import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def function_2(x):
    return x[0]**2 + x[1]**2

fig = plt.figure(figsize=(8,6))
ax3d = plt.axes(projection="3d")

xdata = np.arange(-3.0, 3.0, 0.1)
ydata = np.arange(-3.0, 3.0, 0.1)
X,Y = np.meshgrid(xdata,ydata)
Z = function_2(np.array([X,Y]))

ax3d = plt.axes(projection='3d')
ax3d.plot_surface(X, Y, Z)

ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')

plt.savefig("img.png")