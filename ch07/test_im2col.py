# coding: utf-8
import os
import sys
import numpy as np
sys.path.append("/home/ubuntu/workspace/deep-learning-from-scratch")
from common.util import im2col

x1 = np.random.rand(1, 3, 7, 7)
coll = im2col(x1, 5, 5, stride=1, pad=0)
print(coll.shape)

x2 = np.random.rand(10, 3, 7, 7)
coll2 = im2col(x2, 5, 5, stride=1, pad=0)
print(coll2.shape)
