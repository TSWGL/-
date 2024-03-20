# -*- codeing = utf-8 -*-
# @Time : 2024-02-17 8:13
# @Author : 吴国林
# @File : 7_MeasureOfIndividualConsensusLevel.py
# @Software : PyCharm
import numpy as np

# 已知量与数据
m = 8    # 子群（决策单元）的数量
n = 4    # 待选择或排序的备选方案的数量
tau = 3     # 语言术语集
M = list(range(m))  # 生成表示各个子群（决策单元）的索引列表
N = list(range(n))  # 生成表示各个备选方案的索引列表
T = list(range(2 * tau + 1))    # [-2, -1, 0, 1, 2] --> [0, 1, 2, 3, 4]
lscl = 0.90   # 大群体共识水平阈值
data = np.loadtxt('All_Subgroups_DLPRs.txt')
data1 = np.loadtxt('Consensus_adjusted_DLPRs.txt')
print(data1)
# 查看原始偏好数据

beta_no_3 = np.reshape(data, (m, n, n, (2 * tau) + 1))     # 所有子群的偏好矩阵组成的多维数组：(m,n,n)
print("全部原始偏好:\n", beta_no_3)
inp = list(data1)
print(f"inp\n", inp)
beta3 = beta_no_3[2, :, :, :]
print(f"子群3的偏好：\n", beta3)
for i in N:
    for j in N:
        for t in T:
            inp.append(beta3[i, j, t])  # 第三个子群的偏好数据在列表inp的末尾

beta = np.reshape(inp, (m, n, n, (2 * tau) + 1))
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
    if cl[k] < lscl:
        M1.append(k)
    else:
        M2.append(k)

print(cl, M1, M2)

GCL = sum(cl.values())/m
print(GCL)

with open('All_Consensus_adjusted_DLPRs.txt', 'w') as file:
    for k in inp:
        file.write(str(k) + '\n')  # 将数组元素逐行写入txt 文件，
        # 依次为子群 0, 1, 3, 4, 5, 6, 7, 2 --> 1, 2, 4, 5, 6, 7, 8, 3
# 关闭文件对象
file.close()