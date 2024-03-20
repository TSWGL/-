# -*- codeing = utf-8 -*-
# @Time : 2024-02-27 12:35
# @Author : 吴国林
# @File : 10_Large_group_DLPRs.py
# @Software : PyCharm
import numpy as np
import pickle


data_1 = np.loadtxt('All_Consensus_adjusted_DLPRs.txt')
m = 8
n = 4
tau_1 = 3
M = [0, 1, 3, 4, 5, 6, 7, 2]
N = list(range(n))
T1 = list(range((2 * tau_1) + 1))
beta = np.reshape(data_1, (m, n, n, (2 * tau_1) + 1))

with open("All_Subgroups_social_trust.pkl", 'rb') as f:
    tdc_dic = pickle.load(f)
print(f"子群间的社会信任关系：\n", tdc_dic)    # 子群间的社会信任程度


degree = {}
weights = {}
for k in M:
    # 0 1 3 4 5 6 7 2
    degree[k] = (1 / (len(M) - 1)) * sum(tdc_dic[h, k] for h in M if h != k)
total = sum(degree.values())
for k in degree.keys():
    weights[k] = degree[k] / total
print(weights)

beta_lsg = []
for i in N:
    for j in N:
        for t in T1:
            beta_lsg.append(round(sum(weights[k] * beta[k, i, j, t] for k in M), 2))

res = np.reshape(beta_lsg, (n, n, (2 * tau_1) + 1))
print(res)

priorities = {}
for i in N:
    priorities[i] = round((2 / (n * n)) * sum(
        ((sum((t - tau_1) * res[i, j, t] for t in T1) + tau_1) / (2 * tau_1))
        for j in N), 5)
print(priorities)
print(sum(priorities.values()))