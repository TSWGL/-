# -*- codeing = utf-8 -*-
# @Time : 2024-02-20 13:23
# @Author : 吴国林
# @File : 6_ComputingSubGroupDlprs.py
# @Software : PyCharm
import numpy as np

data_1 = np.loadtxt('Consistent_20_DLPRs.txt')
m = 20
n = 4
tau_1 = 3
N = list(range(n))
T1 = list(range((2 * tau_1) + 1))
beta = np.reshape(data_1, (m, n, n, (2 * tau_1) + 1))

data_2 = np.loadtxt('Social_trust_matrix_2.txt')
tau_2 = 1
T2 = list(range((2 * tau_2) + 1))
td = np.reshape(data_2, (m, m, (2 * tau_2) + 1))


C = {'C_1': [0],
     'C_2': [1],
     'C_3': [2],
     'C_4': [3],
     'C_5': [4],
     'C_6': [5],
     'C_7': [6, 13, 14, 15],
     'C_8': [7, 8, 9, 10, 11, 12, 16, 17, 18, 19]
     }


def func_degree_weight(C_k):
    degree = {}
    weights = {}
    if len(C_k) > 1:
        for k in C_k:
            degree[k] = (1 / (len(C_k) - 1)) * \
                        sum(
                            (((sum((t2 - tau_2) * td[k, h, t2] for t2 in T2)) + tau_2) / (2 * tau_2))
                            for h in C_k
                            if h != k
                        )
    else:
        for k in C_k:
            degree[k] = 1
    total = sum(degree.values())
    for k in degree.keys():
        weights[k] = degree[k] / total
    return weights, degree


def func_subgroup_dlprs(C_k, weights_k):
    beta_sg = []
    for i in N:
        for j in N:
            for t in T1:
                beta_sg.append(
                    round(sum((weights_k[k] * beta[k, i, j, t]) for k in C_k), 2)
                )
    return beta_sg


if __name__ == '__main__':
    beta_sg_matrix = []
    for p in C.keys():
        weights_p, degrees_p = func_degree_weight(C[p])
        beta_sg_p = func_subgroup_dlprs(C[p], weights_p)
        for i in beta_sg_p:
            beta_sg_matrix.append(i)
        print(f"子群{p}的语言分布偏好关系：\n", np.reshape(beta_sg_p, (n, n, 2 * tau_1 + 1)))
        print(f"子群{p}中各个决策者的度中心性：\n", degrees_p)
        print(f"子群{p}中各个决策者的话语权重：\n", weights_p)

    with open('All_Subgroups_DLPRs.txt', 'w') as file:
        for item in beta_sg_matrix:
            file.write(str(item) + '\n')
    file.close()
