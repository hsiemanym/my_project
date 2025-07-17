

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# GPU 설정
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 전체 파이프라인 실행: test 이미지 > 유사도 점수 + 설명


import torch
import tensorflow as tf
print(torch.rand(5, 3))
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available())
