# -*- codeing = utf-8 -*-
# @Time : 2024-02-16 16:41
# @Author : 吴国林
# @File : 4_Clustering.py
# @Software : PyCharm
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "FangSong"  # 设置字体
mpl.rcParams["axes.unicode_minus"] = False  # 正常显示负号

m = 20  # 20个决策者
n = 4  # 4个备选方案
tau = 3
epsilon = 0.96  # 设置子群内部的共识水平阈值
data = np.loadtxt('Consistent_20_DLPRs.txt')  # 加载数据
beta = np.reshape(data, (m, n, n, (2 * tau) + 1))  # 转换数据结构
labels = ['e_1', 'e_2', 'e_3', 'e_4', 'e_5', 'e_6', 'e_7', 'e_8', 'e_9', 'e_10', 'e_11', 'e_12', 'e_13', 'e_14', 'e_15',
          'e_16', '_e17', 'e_18', 'e_19', 'e_20']  # 数据标签
E = list(range(m))  # 构造决策者下标集合
C = {}  # 所有子群构成的集合 C.
z = m  # 参数 z 用于控制迭代过程, 它表示每次迭代开始时的子群个数, 从 z = m 开始, 每次迭代后子群个数减少一个.
num = m  # 参数 num 用于表示新组成的子群的下标, 最开始有 m 个子群, 每个决策者单独是一个子群.

# 初始化。循环表示出 z = m 个子群：子群 0、子群 1、...、子群 m-1. C = {0:[0], 1:[1], ..., m-1:[m-1]}
for k in range(z):
    # 赋予子群 C[K]的决策者的下标
    C[k] = [k]

dn = []  # 用来存放每一次迭代的结果，便于最后使用 scipy.cluster.hierarchy.dendrogram 画图

while z > 1:
    cd_lis = {}
    for p1 in C.keys():
        for p2 in C.keys():
            if p1 < p2:
                temp_cluster = C[p1] + C[p2]  # 合并子群 C[p1]和子群 C[p2] 中的决策者的下标
                cd = (1 / ((len(C[p1]) + len(C[p2])) * (len(C[p1]) + len(C[p2]) - 1) * n * (n - 1) * tau)) * \
                     sum(
                         abs(
                             sum(
                                 (t - tau) * beta[k, i, j, t]
                                 for t in range((2 * tau) + 1)
                             )
                             -
                             sum(
                                 (t - tau) * beta[h, i, j, t]
                                 for t in range((2 * tau) + 1)
                             )
                         )
                         for k in temp_cluster
                         for h in temp_cluster
                         for i in range(n)
                         for j in range(n)
                         if h != k and i < j
                     )
                cd_lis[(str(p1), str(p2))] = cd
    print(z, cd_lis)
    min_key = min(cd_lis.items(), key=lambda x: x[1])[0]
    min_value = min(cd_lis.items(), key=lambda x: x[1])[1]
    left, right = int(min_key[0]), int(min_key[1])
    # 把共识度最高的两个子群里的决策者放在一起组成一个新的子群
    # Partition[(min_key[0], min_key[1])] = Partition[left] + Partition[right]
    C[num] = C[left] + C[right]
    del C[left]
    del C[right]
    dn.append([left, right, min_value, len(C[num])])
    # print(dn)
    num += 1
    z -= 1

print(np.reshape(dn, (-1, 4)))

# Plot dendrogram
x = range(0, 20, 1)
plt.figure(figsize=(12, 5))
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Decision makers')
# plt.ylabel('Cohesion')
# plt.xticks(list(x)[::1], labels[::1], fontsize=25, rotation=45)
# plt.yticks(fontsize=20)
plt.title('20个决策者调整后的语言分布偏好关系的聚类树图', size=25)
plt.xlabel('决策者', size=25)
plt.ylabel('子群内部的冲突程度', size=25)
plt.grid(alpha=0.25)  # alpha 透明度，当alpha=1表示完全不透明
dendrogram(np.reshape(dn, (-1, 4)), labels=np.array(labels))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# 添加横向参考线及其标签
plt.axhline(y=1 - epsilon, color="green", linestyle="--")
plt.text(0, 1 - epsilon + 0.002, 'y = 0.04', fontsize=25, color="green", horizontalalignment='left')
plt.show()
