# -*- codeing = utf-8 -*-
# @Time : 2024-02-19 9:49
# @Author : 吴国林
# @File : 2_ComputingCardinalConsistencyLevelOfDlprs.py
# @Software : PyCharm
import numpy as np

# data = np.loadtxt('Original_DLPRs_4.txt')  # 数据记录的是: 第4个决策者的原始 beta_[k,i,j,t]
data = np.loadtxt('Consistent_20_DLPRs.txt')

# with open("Consistent_20_DLPRs.txt", "r") as f:  # 打开文件
#     data = f.read()  # 读取文件
#     print(data)

m = 20
n = 4    # 待选择或排序的备选方案的数量
tau = 3  # LTS的表示 2 * tau + 1 = 7
N = list(range(n))  # 生成表示各个备选方案的索引列表
T = list(range((2 * tau) + 1))    # 生成表示各个语言术语的索引列表 T = [-2, -1, 0, 1, 2] --> [0, 1, 2, 3, 4]

B = np.reshape(data, (m, n, n, (2 * tau) + 1))    # beta[i,j,t]


def func_aci(beta):
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
    return aci


if __name__ == '__main__':
    aci_dict = {}
    for k in range(m):
        beta_k = B[k, :, :, :]
        aci_k = func_aci(beta_k)
        aci_dict[k] = round(aci_k, 3)
    print(aci_dict)
    with open('Consistent_20_DLPRs_aci.txt', 'w') as file:
        for item in aci_dict.keys():
            file.write(str(aci_dict[item]) + '\n')  # 将数组元素逐行写入txt 文件
    # 关闭文件对象
    file.close()
