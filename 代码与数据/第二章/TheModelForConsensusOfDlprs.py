# -*- codeing = utf-8 -*-
# @Time : 2024-02-10 16:28
# @Author : 吴国林
# @File : TheModelForConsensusOfDlprs.py
# @Software : PyCharm
import numpy as np
import xlwt
import gurobipy as gp

# 已知量与数据
data = np.loadtxt('50_DLPRs.txt')  # 数据记录的是: beta_[k,i,j,t]
m = 50    # 决策者的数量
n = 4    # 待选择或排序的备选方案的数量
tau = 2  # LTS的表示 2 * tau + 1 = 5
M = list(range(m))  # 生成表示各个决策者的索引列表
N = list(range(n))  # 生成表示各个备选方案的索引列表
T = list(range(2 * tau + 1)) # 生成表示各个语言术语的索引列表
M1 = []
M2 = []

# 查看原始偏好数据
B = np.reshape(data, (m, n, n, 2 * tau + 1)) # data[k,i,j,t]
b0 = B[0, :, :, :]     # 取某个决策者的 DLPR
# print("决策者0的原始偏好:\n", b0)
# 设置模型参数的初始值
eta = 0.8   # 根据特定决策问题预先给定的想要达成的个体的加型基数一致性水平
kM = 10     # 一个较大的正数，用于0-1整数规划建模
ke = 0.0001  # 一个较小的正数，用于将严格不等式转化为基本不等式

def model1(b):
    # 创建决策变量
    model = gp.Model()
    # 创建决策变量
    beta = {}      # 一个用于存放变量 beta_[k,i,j,t] (k in S, i<j) 的字典
    u = {}      # 一个用于存放变量 u_k,i,j (k in S, i<j) 的字典
    v = {}      # 一个用于存放变量 v_k,i,j (k in S, i<j) 的字典
    delta = {}  # 一个用于存放变量 delta_k,i,j (k in S, i<j) 的字典，用来线性化目标函数里的绝对值符号
    f = {}      # 一个用于存放变量 f_k,i,e,j (k in S, i<e<j) 的字典

    for k in M:
        for i in N:
            for j in N:
                for t in T:
                    beta[k, i, j, t] = model.addVar(lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name=f"beta[{k},{i},{j},{t}]")

    model.update()
    obj = gp.LinExpr()
    for k in M:
        for i in N:
            for j in N:
                if i < j:
                    obj += delta[i, j]
    model.setObjective(obj, sense=gp.GRB.MINIMIZE)