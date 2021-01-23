# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from collections import OrderedDict

from functions import *
from layer_naive import *
from two_layer_net  import *
from optimizer import *

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# hyper prameter
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1) #600
input_size=784
hidden_size=50
output_size=10

# He initial value
weight_init_std=np.sqrt(2.0 / (input_size + hidden_size + output_size))

network = TwoLayerNet(input_size, hidden_size, output_size, weight_init_std)
optimizer = SGD(learning_rate)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grads = network.gradient(x_batch, t_batch)
    
    optimizer.update(network.params, grads)
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    #calc per epoch accuracy
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    
# show loss image
fig = plt.figure()

plt.plot(train_loss_list)
plt.xlabel("iter_num")
plt.ylabel("loss")

fig.savefig("train_loss_img.png")

# show epoch image
fig = plt.figure()

plt.plot(train_acc_list, label="train_acc")
plt.plot(test_acc_list, linestyle="--", label="test_acc")
plt.xlabel("epochs")
plt.ylabel("accuracy")

fig.savefig("epoch_img.png")
