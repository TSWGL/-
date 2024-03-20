# -*- codeing = utf-8 -*-
# @Time : 2024-02-10 16:28
# @Author : 吴国林
# @File : TheModelForConsensusOfDlprs.py
# @Software : PyCharm
import numpy as np
import gurobipy as gp

# 1.1 已知量与数据
data = np.loadtxt('4_consistency_DLPRs.txt')  # 数据记录的是: beta_[k,i,j,t], 均满足一致性要求的DLPRs
m = 4    # 决策者的数量
n = 4    # 待选择或排序的备选方案的数量
tau = 2  # LTS的表示 2 * tau + 1 = 5
M = list(range(m))  # 生成表示各个决策者的索引列表
N = list(range(n))  # 生成表示各个备选方案的索引列表
T = list(range(2 * tau + 1))    # 生成表示各个语言术语的索引列表 T = [-3, -2, -1, 0, 1, 2, 3] --> [0, 1, 2, 3, 4, 5, 6]
# T = [-3, -2, -1, 0, 1, 2, 3] --> [0, 1, 2, 3, 4, 5, 6]
T1 = list(range(2 * tau))   # 剔除 tau
print(T, T1)
M1 = [0]     # 根据个体共识水平确定低于群体共识水平阈值的决策者的下标,由 MeasureOfIndividualConsensusLevel.py 给出
M2 = [1, 2, 3]     # 根据个体共识水平确定大于等于群体共识水平阈值的决策者的下标,由 MeasureOfIndividualConsensusLevel.py 给出

# 1.2 查看原始偏好数据
beta = np.reshape(data, (m, n, n, 2 * tau + 1))    # B[k,i,j,t]
beta0 = beta[0, :, :, :]     # 需要手动确定k的值, 取某个决策者的 DLPRs
print(beta0)
# print("决策者0的原始偏好:\n", b0)
# 设置模型参数的初始值
aci2 = 0.8   # 根据特定决策问题预先给定的想要达成的个体的加型基数一致性水平
KM = 10     # 一个较大的正数，用于0-1整数规划建模
ke = 0.001  # 一个较小的正数，用于将严格不等式转化为基本不等式
z1 = {0: 1, 1: 1, 2: 1, 3: 1}  # beta中语言术语最小的个数，待手动填写
z2 = {0: 2, 1: 2, 2: 2, 3: 2}  # beta中语言术语最多的个数，待手动填写
gcl = 0.8   # 群体共识水平阈值

# 中间量计算
ns = {}  # 计算集合M2中决策者的语言分布评估的数值标度
for q in M2:
    for i in N:
        for j in N:
            ns[q, i, j] = sum((t - tau) * beta[q, i, j, t] for t in T)
print(ns)
xi = {}
for q in M2:
    for k in M2:
        for i in N:
            for j in N:
                if k != q:
                    xi[q, k, i, j] = abs(ns[q, i, j] - ns[k, i, j])
print(xi)

# 1.3 创建模型
model = gp.Model()

# 1.4 创建决策变量
beta1 = {}      # 一个用于存放变量 beta1_[k,i,j,t] 的字典, k in M1
x = {}  # 一个用于存放变量 u_[k,i,j] 的字典, k in M1
y = {}  # 一个用于存放变量 v_[k,i,j] 的字典, k in M1
d = {}  # 一个用于存放变量 delta_[k,i,j,t] 的字典，用来线性化目标函数里的绝对值符号, k in M1
fai = {}  # 一个用于存放变量 fai_[k,i,j,t] 的字典，用来线性化目标函数里的绝对值符号, k in M1
o1 = {}  # 一个用于存放变量 o_[k,i,j,t] 的字典,用于保证为类FLPRs的DLPRs, k in M1
f = {}  # 一个用于存放变量 f_[k,i,h,j] 的字典,用于线性化控制加型基数一致性的约束, k in M1
gamma = {}  # 一个用于存放变量 gamma_[k,i,j,t] 的字典,用于线性化控制加型基数一致性的约束, k in M1
r1 = {}  # 一个用于存放变量 r_[k,i,j] 的字典,用于表示一个(i,j)中，beta1_[k,i,j,t]不为0的个数, k in M1
ns1 = {}  # 数值标度

kai = {}
luo = {}
pai = {}
# xi = {}

# 变量 beta1[p, i, j, t], p in M1
for p in M1:
    for i in N:
        for j in N:
            for t in T:
                beta1[p, i, j, t] = model.addVar(lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name=f"beta1[{p},{i},{j},{t}]")
# 变量 d[k, i, j, t], k in M1
for p in M1:
    for i in N:
        for j in N:
            for t in T:
                d[p, i, j, t] = model.addVar(lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name=f"d[{p},{i},{j},{t}]")
# 变量 o1[k, i, j, t], k in M1
for p in M1:
    for i in N:
        for j in N:
            for t in T:
                o1[p, i, j, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"o1[{p},{i},{j},{t}]")  # 0-1变量，测试需要上下界吗
# 变量 kai[k,i, j, t], k in M1
for p in M1:
    for i in N:
        for j in N:
            for t in T:
                fai[p, i, j, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"fai[{p},{i},{j},{t}]")  # 0-1变量，测试需要上下界吗
# 变量 gamma[k, i, j, t], k in M1
for p in M1:
    for i in N:
        for j in N:
            for t in T:
                gamma[p, i, j, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"gamma[{p},{i},{j},{t}]")  # 变量取值范围？
# 变量 f[k, i, h, j], k in M1
for p in M1:
    for i in N:
        for h in N:
            for j in N:
                f[p, i, h, j] = model.addVar(lb=-3 * tau, ub=3 * tau, vtype=gp.GRB.CONTINUOUS,
                                             name=f"f[{p},{i},{h},{j}]")
# 变量 x[k, i, j], k in M1
for p in M1:
    for i in N:
        for j in N:
            x[p, i, j] = model.addVar(lb=0, ub=1, vtype=gp.GRB.BINARY, name=f"x[{p},{i},{j}]")  # 0-1变量，测试需要上下界吗
# 变量 y[k, i, j], k in M1
for p in M1:
    for i in N:
        for j in N:
            y[p, i, j] = model.addVar(lb=0, ub=1, vtype=gp.GRB.BINARY, name=f"y[{p},{i},{j}]")  # 0-1变量，测试需要上下界吗
# 变量 r1[k, i, j], k in M1
for p in M1:
    for i in N:
        for j in N:
            r1[p, i, j] = model.addVar(vtype=gp.GRB.INTEGER, name=f"r1[{p},{i},{j}]")    # 整数变量
# 变量 ns1[p, i, j], k in M1
for p in M1:
    for i in N:
        for j in N:
            ns1[p, i, j] = model.addVar(lb=-tau, ub=tau, vtype=gp.GRB.CONTINUOUS, name=f"ns1[{p},{i},{j}]")
# 变量 kai[p, k, i, j], p, k in M1; p != k
for p in M1:
    for k in M1:
        for i in N:
            for j in N:
                if k != p:
                    kai[p, k, i, j] = model.addVar(lb=0, ub=2*tau, vtype=gp.GRB.CONTINUOUS, name=f"kai[{p},{k},{i},{j}")
# 变量 luo[p, k, i, j], p in M1; k in M2;
for p in M1:
    for k in M2:
        for i in N:
            for j in N:
                luo[p, k, i, j] = model.addVar(lb=0, ub=2*tau, vtype=gp.GRB.CONTINUOUS, name=f"luo[{p},{k},{i},{j}")
# 变量 pai[q, k, i, j], q in M2; k in M1
for q in M2:
    for k in M1:
        for i in N:
            for j in N:
                pai[q, k, i, j] = model.addVar(lb=0, ub=2*tau, vtype=gp.GRB.CONTINUOUS, name=f"pai[{q},{k},{i},{j}")


# 1.5 模型更新
model.update()

# 1.6 创建目标函数
obj = gp.LinExpr()
for p in M1:
    for i in N:
        for j in N:
            for t in T:
                obj += fai[p, i, j, t]
model.setObjective((1 / 2) * obj, sense=gp.GRB.MINIMIZE)

# 1.7 循环创建约束
# '1-1'
for p in M1:
    for i in N:
        for j in N:
            for t in T:
                model.addConstr(d[p, i, j, t] == beta1[p, i, j, t] - beta[p, i, j, t], name=f"c1[{p},{i},{j},{t}]")
                # 等式约束
# '1-2'
for p in M1:
    for i in N:
        for j in N:
            for t in T:
                model.addConstr(d[p, i, j, t] - KM * fai[p, i, j, t] <= 0, name=f"c2[{p},{i},{j},{t}]")
# '1-3'
for p in M1:
    for i in N:
        for j in N:
            for t in T:
                model.addConstr(-d[p, i, j, t] - KM * fai[p, i, j, t] <= 0, name=f"c3[{p},{i},{j},{t}]")
# '1-4'
for p in M1:
    for i in N:
        for j in N:
            for t in T:
                if i != j:
                    model.addConstr(beta1[p, j, i, t] - beta1[p, i, j, (2 * tau) - t] == 0, name=f"c4[{p},{i},{j},{t}]")
# '1-5'
for p in M1:
    for i in N:
        model.addConstr(beta1[p, i, i, tau] == 1, name=f"c6[{p},{i}]")  # 等式约束
# '1-6'
for p in M1:
    for i in N:
        for j in N:
            model.addConstr(sum(beta1[p, i, j, t] for t in T) == 1, name=f"c5[{p},{i},{j}]")    # 等式约束
# 约束'1-7'是定义变量beta1的上下限，已经在定义变量时完成.
# '1-8'
for p in M1:
    for i in N:
        for j in N:
            model.addConstr(r1[p, i, j] == sum(o1[p, i, j, t] for t in T), name=f"c8[{p},{i}]")  # 等式约束
# '1-9'
for p in M1:
    for i in N:
        for j in N:
            for t in T:
                model.addConstr(o1[p, i, j, t] - beta1[p, i, j, t] <= 1 - ke, name=f"c9[{p},{i},{j},{t}]")
# '1-10'
for p in M1:
    for i in N:
        for j in N:
            for t in T:
                model.addConstr(-o1[p, i, j, t] + beta1[p, i, j, t] <= 0, name=f"c10[{p},{i},{j},{t}]")
# '1-11'
for p in M1:
    for i in N:
        for j in N:
            if i != j:
                model.addConstr(r1[p, i, j] - z1[p] >= 0, name=f"c11-1[{p},{i},{j}]")
                model.addConstr(r1[p, i, j] - z2[p] <= 0, name=f"c11-2[{p},{i},{j}]")
# '1-12'
for p in M1:
    for i in N:
        for j in N:
            model.addConstr(sum(gamma[p, i, j, t] for t in T1) <= 2, name=f"c12[{p},{i},{j}]")
# '1-13'
for p in M1:
    for i in N:
        for j in N:
            for t in T1:
                model.addConstr(o1[p, i, j, t + 1] - o1[p, i, j, t] <= gamma[p, i, j, t], name=f"c13[{p},{i},{j},{t}]")
# '1-14'
for p in M1:
    for i in N:
        for j in N:
            for t in T1:
                model.addConstr(-o1[p, i, j, t + 1] + o1[p, i, j, t] <= gamma[p, i, j, t], name=f"c14[{p},{i},{j},{t}]")
# '1-15'
for p in M1:
    for i in N:
        for j in N:
            model.addConstr(o1[p, i, j, 0] + o1[p, i, j, 2 * tau] <= 1, name=f"c15[{p},{i},{j}]")
# '1-16'
for p in M1:
    for i in N:
        for j in N:
            model.addConstr(ns1[p, i, j] + KM * (1 - x[p, i, j]) - ke >= 0, name=f"c16[{p},{i},{j}]")
# '1-17'
for p in M1:
    for i in N:
        for j in N:
            model.addConstr(ns1[p, i, j] - KM * x[p, i, j] <= 0, name=f"c17[{p},{i},{j}]")
# '1-18'
for p in M1:
    for i in N:
        for j in N:
            model.addConstr(ns1[p, i, j] + KM * (1 - y[p, i, j]) >= 0, name=f"c18[{p},{i},{j}]")
# '1-19'
for p in M1:
    for i in N:
        for j in N:
            model.addConstr(ns1[p, i, j] - KM * (1 - y[p, i, j]) <= 0, name=f"c19[{p},{i},{j}]")
# '1-20'
for p in M1:
    for i in N:
        for j in N:
            model.addConstr(ns1[p, i, j] - KM * (x[p, i, j] + y[p, i, j]) <= 0 - ke, name=f"c20[{p},{i},{j}]")
# '1-21'
for p in M1:
    for i in N:
        for j in N:
            model.addConstr(x[p, i, j] + y[p, i, j] <= 1, name=f"c21[{p},{i},{j}]")
# '1-22'
for p in M1:
    for i in N:
        for h in N:
            for j in N:
                model.addConstr(ns1[p, i, j] + KM * (2 - x[p, i, h] - x[p, h, j]) >= 0 + ke,
                                name=f"c22[{p},{i},{h},{j}]")
# '1-23'
for p in M1:
    for i in N:
        for h in N:
            for j in N:
                model.addConstr(ns1[p, i, j] + KM * (2 - y[p, i, h] - y[p, h, j]) >= 0, name=f"c23[{p},{i},{h},{j}]")
# '1-24'
for p in M1:
    for i in N:
        for h in N:
            for j in N:
                model.addConstr(ns1[p, i, j] - KM * (2 - y[p, i, h] - y[p, h, j]) <= 0, name=f"c24[{p},{i},{h},{j}]")
# '1-25'
for p in M1:
    for i in N:
        for h in N:
            for j in N:
                model.addConstr(ns1[p, i, j] + KM * (2 - x[p, i, h] - y[p, h, j]) >= 0 + ke,
                                name=f"c25[{p},{i},{h},{j}]")
# '1-26'
for p in M1:
    for i in N:
        for h in N:
            for j in N:
                model.addConstr(ns1[p, i, j] + KM * (2 - y[p, i, h] - x[p, h, j]) >= 0 + ke,
                                name=f"c26[{p},{i},{h},{j}]")
# '1-27'
for p in M1:
    for i in N:
        for h in N:
            for j in N:
                model.addConstr(ns1[p, i, h] + ns1[p, h, j] - ns1[p, i, j] <= f[p, i, h, j],
                                name=f"c27[{p},{i},{h},{j}]")
# '1-28'
for p in M1:
    for i in N:
        for h in N:
            for j in N:
                model.addConstr(-ns1[p, i, h] - ns1[p, h, j] + ns1[p, i, j] <= f[p, i, h, j],
                                name=f"c28[{p},{i},{h},{j}]")
# '1-29'
model.addConstr(sum(f[p, i, h, j]
                    for p in M1
                    for i in N
                    for h in N
                    for j in N
                    if i < h < j)
                <= (1 - aci2) * ((tau * n * (n-1) * (n-2)) / 2), name="c29")
# '1-30' 已经在定义变量的时候进行了约束
# '1-31'
for p in M1:
    for i in N:
        for j in N:
            model.addConstr(ns1[p, i, j] == sum(((t - tau) * beta1[p, i, j, t]) for t in T), name=f"c31[{p},{i},{j}]")
# '1-32' 已经在定义变量的时候进行了约束
# '1-33'
for p in M1:
    model.addConstr(sum(kai[p, k, i, j]
                        for k in M1
                        for i in N
                        for j in N
                        if k != p
                        )
                    +
                    sum(luo[p, k, i, j]
                        for k in M2
                        for i in N
                        for j in N
                        )
                    <= tau * n * (n-1) * (m - 1) * (1 - gcl), name=f"c33[{p}]")
# '1-34'
for p in M1:
    for k in M1:
        for i in N:
            for j in N:
                if p != k:
                    model.addConstr(ns1[p, i, j] - ns1[k, i, j] <= kai[p, k, i, j], name=f"c34[{p},{k},{i},{j}]")
# '1-35'
for p in M1:
    for k in M1:
        for i in N:
            for j in N:
                if p != k:
                    model.addConstr(-ns1[p, i, j] + ns1[k, i, j] <= kai[p, k, i, j], name=f"c35[{p},{k},{i},{j}]")
# '1-36'
for p in M1:
    for k in M2:
        for i in N:
            for j in N:
                model.addConstr(ns1[p, i, j] - ns[k, i, j] <= luo[p, k, i, j], name=f"c36[{p},{k},{i},{j}]")
# '1-37'
for p in M1:
    for k in M2:
        for i in N:
            for j in N:
                model.addConstr(-ns1[p, i, j] + ns[k, i, j] <= luo[p, k, i, j], name=f"c37[{p},{k},{i},{j}]")
# '1-38'
for q in M2:
    model.addConstr(sum(pai[q, k, i, j]
                        for k in M1
                        for i in N
                        for j in N
                        )
                    +
                    sum(xi[q, k, i, j]
                        for k in M2
                        for i in N
                        for j in N
                        if k != q
                        )
                    <= tau * n * (n-1) * (m - 1) * (1 - gcl), name=f"c38[{q}]")
# '1-39'
for q in M2:
    for k in M1:
        for i in N:
            for j in N:
                model.addConstr(ns[q, i, j] - ns1[k, i, j] <= pai[q, k, i, j], name=f"c39[{q},{k},{i},{j}]")
# '1-40'
for q in M2:
    for k in M1:
        for i in N:
            for j in N:
                model.addConstr(-ns[q, i, j] + ns1[k, i, j] <= pai[q, k, i, j], name=f"c40[{q},{k},{i},{j}]")

# 1.8 编译具体模型
model.write('Model3.lp')

# 1.9 求解
model.optimize()
x_matrix = []
if model.Status == gp.GRB.OPTIMAL:
    for k in M1:
        for i in N:
            for j in N:
                for t in T:
                    x_matrix.append(beta1[k, i, j, t].X)
                    print(f"[{k},{i},{j},{t}]:", beta1[k, i, j, t].X)
modified_p = np.reshape(np.array(x_matrix), (len(M1), n, n, (2 * tau) + 1))
print("M1中所有决策者的满足次序一致性要求的修正偏好矩阵:\n", modified_p)