

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import tensorflow as tf
print(torch.rand(5, 3))
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available())