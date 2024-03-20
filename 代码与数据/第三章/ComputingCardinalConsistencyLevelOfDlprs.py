# -*- codeing = utf-8 -*-
# @Time : 2024-02-19 9:49
# @Author : 吴国林
# @File : ComputingCardinalConsistencyLevelOfDlprs.py
# @Software : PyCharm
import numpy as np

# data = np.loadtxt('Original_DLPRs_4.txt')  # 数据记录的是: 第4个决策者的原始 beta_[k,i,j,t]
data = np.loadtxt('OrdinalConsistency_DLPRs_4_1.txt')
n = 4    # 待选择或排序的备选方案的数量
tau = 2  # LTS的表示 2 * tau + 1 = 5
N = list(range(n))  # 生成表示各个备选方案的索引列表
T = list(range((2 * tau) + 1))    # 生成表示各个语言术语的索引列表 T = [-2, -1, 0, 1, 2] --> [0, 1, 2, 3, 4]

beta = np.reshape(data, (n, n, (2 * tau) + 1))    # beta[i,j,t]
print(beta)

aci = 1 - (2 / (tau * n * (n-1) * (n-2))) * sum(
    abs(
        sum((t - tau) * beta[i, h, t] for t in T) +
        sum((t - tau) * beta[h, j, t] for t in T) -
        sum((t - tau) * beta[i, j, t] for t in T)
    )
    for i in N
    for h in N
    for j in N
    if i < h < j
)

print(aci)