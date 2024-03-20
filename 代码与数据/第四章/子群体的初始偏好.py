# -*- codeing = utf-8 -*-
# @Time : 2024-02-29 13:56
# @Author : 吴国林
# @File : 子群体的初始偏好.py
# @Software : PyCharm
import numpy as np

data = np.loadtxt('All_Subgroups_DLPRs.txt')
m = 8
n = 4
tau = 3
beta = np.reshape(data, (m, n, n, 2 * tau + 1))
for k in range(8):
    print(f"第{k+1}个子群的偏好为：\n", beta[k, :, :, :])