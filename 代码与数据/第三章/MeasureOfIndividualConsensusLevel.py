# -*- codeing = utf-8 -*-
# @Time : 2024-02-17 8:13
# @Author : 吴国林
# @File : MeasureOfIndividualConsensusLevel.py
# @Software : PyCharm
import numpy as np

# 已知量与数据
m = 4    # 子群（决策单元）的数量
n = 4    # 待选择或排序的备选方案的数量
tau = 2     # 语言术语集
M = list(range(m))  # 生成表示各个子群（决策单元）的索引列表
N = list(range(n))  # 生成表示各个备选方案的索引列表
T = list(range(2 * tau + 1))    # [-2, -1, 0, 1, 2] --> [0, 1, 2, 3, 4]
scl = 0.8   # 群体共识水平阈值
data = np.loadtxt('4_consensus_DLPRs.txt')

# 查看原始偏好数据
beta = np.reshape(data, (m, n, n, (2 * tau) + 1))     # 所有子群的偏好矩阵组成的多维数组：(m,n,n)
print("全部原始偏好:\n", beta)

cl = {}
for k in M:
    cl[k] = round(1 - ((1 / ((m - 1) * n * (n - 1) * tau)) *
                  sum(
                      abs(
                          sum(
                              (t - tau) * beta[k, i, j, t]
                              for t in T
                          )
                              -
                          sum(
                          (t - tau) * beta[h, i, j, t]
                              for t in T
                          )
                      )
                      for h in M
                      for i in N
                      for j in N
                      if h != k and i < j
                  )
                       )
                  , 3)

M1 = []
M2 = []
for k in M:
    if cl[k] < scl:
        M1.append(k)
    else:
        M2.append(k)

print(cl, M1, M2)

GCL = sum(cl.values())/m
print(GCL)