import torch
import torch.nn as nn
import torch.nn
import numpy as np

# 2 * 2 * 3
a = torch.tensor([[[1.0,2.0,3.0],
                 [4.0,5.0,6.0]],
                [[1.0,2.0,3.0],
                 [4.0,5.0,6.0]]])

print("calculated by pytorch:")
ln = nn.LayerNorm(normalized_shape=(a.shape[-2], a.shape[-1]), elementwise_affine=False)
print(ln(a))


print("calculated by numpy:")
# 将axis指定的维度展开，计算均值
mean = np.mean(a.numpy(), axis=(1,2))
# 将axis指定的维度展开，计算方差
var = np.var(a.numpy(), axis=(1,2))
div = np.sqrt(var)+1e-5
# 计算结果
res = (a.numpy()-mean[:,None,None])/div[:,None, None]
print(res)