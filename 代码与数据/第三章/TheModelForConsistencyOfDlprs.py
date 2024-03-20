# -*- codeing = utf-8 -*-
# @Time : 2024-02-10 17:21
# @Author : 吴国林
# @File : TheModelForConsistencyOfDlprs.py
# @Software : PyCharm
import numpy as np
import gurobipy as gp

# 1.1 已知量与数据
data = np.loadtxt('Original_DLPRs_4.txt')  # 数据记录的是: 第4个决策者的原始 beta_[k,i,j,t]
n = 4    # 待选择或排序的备选方案的数量
tau = 2  # LTS的表示 2 * tau + 1 = 5
N = list(range(n))  # 生成表示各个备选方案的索引列表
T = list(range(2 * tau + 1))    # 生成表示各个语言术语的索引列表 T = [-2, -1, 0, 1, 2] --> [0, 1, 2, 3, 4]
T1 = list(range(2 * tau))   # 剔除 tau

# 1.2 查看原始偏好数据
beta = np.reshape(data, (n, n, 2 * tau + 1))    # data[k,i,j,t]
print(beta)

# 设置模型参数的初始值
aci2 = 0.9   # 根据特定决策问题预先给定的想要达成的个体的加型基数一致性水平
KM = 10     # 一个较大的正数，用于0-1整数规划建模
ke = 0.001  # 一个较小的正数，用于将严格不等式转化为基本不等式
z1 = 1  # beta中语言术语最小的个数，待手动填写
z2 = 2  # beta中语言术语最多的个数，待手动填写
# z2 = 3


# 1.3 创建模型
model = gp.Model()

# 1.4 创建决策变量
beta1 = {}      # 一个用于存放变量 beta1_[i,j,t] 的字典
x = {}  # 一个用于存放变量 u_[i,j] 的字典
y = {}  # 一个用于存放变量 v_[i,j] 的字典
d = {}  # 一个用于存放变量 delta_[i,j,t] 的字典，用来线性化目标函数里的绝对值符号
kai = {}  # 一个用于存放变量 delta_[i,j,t] 的字典，用来线性化目标函数里的绝对值符号
o1 = {}  # 一个用于存放变量 o_[i,j,t] 的字典,用于保证为FLPRs
f = {}  # 一个用于存放变量 f_[i,h,j] 的字典,用于线性化控制加型基数一致性的约束
gamma = {}  # 一个用于存放变量 gamma_[i,j,t] 的字典,用于线性化控制加型基数一致性的约束
r1 = {}  # 一个用于存放变量 r_[i,j] 的字典,用于表示一个(i,j)中，beta1_[i,j,t]不为0的个数
ns1 = {}  # 数值标度

# 变量 beta1[i, j, t]
for i in N:
    for j in N:
        for t in T:
            beta1[i, j, t] = model.addVar(lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name=f"beta1[{i},{j},{t}]")
# 变量 d[i, j, t]
for i in N:
    for j in N:
        for t in T:
            d[i, j, t] = model.addVar(lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name=f"d[{i},{j},{t}]")
# 变量 o1[i, j, t]
for i in N:
    for j in N:
        for t in T:
            o1[i, j, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"o1[{i},{j},{t}]")  # 0-1变量，测试需要上下界吗
# 变量 kai[i, j, t]
for i in N:
    for j in N:
        for t in T:
            kai[i, j, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"kai[{i},{j},{t}]")  # 0-1变量，测试需要上下界吗
# 变量 gamma[i, j, t]
for i in N:
    for j in N:
        for t in T:
            gamma[i, j, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"gamma[{i},{j},{t}]")  # 变量取值范围？
# 变量 f[i, h, j]
for i in N:
    for h in N:
        for j in N:
            f[i, h, j] = model.addVar(lb=-3 * tau, ub=3 * tau, vtype=gp.GRB.CONTINUOUS, name=f"f[{i},{h},{j}]")
# 变量 x[i, j]
for i in N:
    for j in N:
        x[i, j] = model.addVar(lb=0, ub=1, vtype=gp.GRB.BINARY, name=f"x[{i},{j}]")  # 0-1变量，测试需要上下界吗
# 变量 y[i, j]
for i in N:
    for j in N:
        y[i, j] = model.addVar(lb=0, ub=1, vtype=gp.GRB.BINARY, name=f"y[{i},{j}]")  # 0-1变量，测试需要上下界吗
# 变量 r1[i, j]
for i in N:
    for j in N:
        r1[i, j] = model.addVar(vtype=gp.GRB.INTEGER, name=f"r1[{i},{j}]")    # 整数变量
# 变量 ns1[i, j]
for i in N:
    for j in N:
        ns1[i, j] = model.addVar(lb=-tau, ub=tau, vtype=gp.GRB.CONTINUOUS, name=f"ns1[{i},{j}]")

# 1.5 模型更新
model.update()

# 1.6 创建目标函数
obj = gp.LinExpr()
for i in N:
    for j in N:
        for t in T:
            obj += kai[i, j, t]
model.setObjective((1 / 2) * obj, sense=gp.GRB.MINIMIZE)

# 1.7 循环创建约束
# '1-1'
for i in N:
    for j in N:
        for t in T:
            model.addConstr(d[i, j, t] == beta1[i, j, t] - beta[i, j, t], name=f"c1[{i},{j},{t}]")       # 等式约束
# '1-2'
for i in N:
    for j in N:
        for t in T:
            model.addConstr(d[i, j, t] - KM * kai[i, j, t] <= 0, name=f"c2[{i},{j},{t}]")
# '1-3'
for i in N:
    for j in N:
        for t in T:
            model.addConstr(-d[i, j, t] - KM * kai[i, j, t] <= 0, name=f"c3[{i},{j},{t}]")
# '1-4'
for i in N:
    for j in N:
        for t in T:
            if i != j:
                model.addConstr(beta1[j, i, t] - beta1[i, j, (2 * tau) - t] == 0, name=f"c4[{i},{j},{t}]")
# '1-5'
for i in N:
    model.addConstr(beta1[i, i, tau] == 1, name=f"c6[{i}]")  # 等式约束
# '1-6'
for i in N:
    for j in N:
        model.addConstr(sum(beta1[i, j, t] for t in T) == 1, name=f"c5[{i},{j}]")    # 等式约束
# 约束'1-7'是定义变量beta1的上下限，已经在定义变量时完成.
# '1-8'
for i in N:
    for j in N:
        model.addConstr(r1[i, j] == sum(o1[i, j, t] for t in T), name=f"c8[{i}]")  # 等式约束
# '1-9'
for i in N:
    for j in N:
        for t in T:
            model.addConstr(o1[i, j, t] - beta1[i, j, t] <= 1 - ke, name=f"c9[{i},{j},{t}]")
# '1-10'
for i in N:
    for j in N:
        for t in T:
            model.addConstr(-o1[i, j, t] + beta1[i, j, t] <= 0, name=f"c10[{i},{j},{t}]")
# '1-11'
for i in N:
    for j in N:
        if i != j:
            model.addConstr(r1[i, j] - z1 >= 0, name=f"c11-1[{i},{j}]")
            model.addConstr(r1[i, j] - z2 <= 0, name=f"c11-2[{i},{j}]")
# '1-12'
for i in N:
    for j in N:
        model.addConstr(sum(gamma[i, j, t] for t in T1) <= 2, name=f"c12[{i},{j}]")
# '1-13'
for i in N:
    for j in N:
        for t in T1:
            model.addConstr(o1[i, j, t + 1] - o1[i, j, t] <= gamma[i, j, t], name=f"c13[{i},{j},{t}]")
# '1-14'
for i in N:
    for j in N:
        for t in T1:
            model.addConstr(-o1[i, j, t + 1] + o1[i, j, t] <= gamma[i, j, t], name=f"c14[{i},{j},{t}]")
# '1-15'
for i in N:
    for j in N:
        model.addConstr(o1[i, j, 0] + o1[i, j, 2 * tau] <= 1, name=f"c15[{i},{j}]")
# '1-16'
for i in N:
    for j in N:
        model.addConstr(ns1[i, j] + KM * (1 - x[i, j]) - ke >= 0, name=f"c16[{i},{j}]")
# '1-17'
for i in N:
    for j in N:
        model.addConstr(ns1[i, j] - KM * x[i, j] <= 0, name=f"c17[{i},{j}]")
# '1-18'
for i in N:
    for j in N:
        model.addConstr(ns1[i, j] + KM * (1 - y[i, j]) >= 0, name=f"c18[{i},{j}]")
# '1-19'
for i in N:
    for j in N:
        model.addConstr(ns1[i, j] - KM * (1 - y[i, j]) <= 0, name=f"c19[{i},{j}]")
# '1-20'
for i in N:
    for j in N:
        model.addConstr(ns1[i, j] - KM * (x[i, j] + y[i, j]) <= 0 - ke, name=f"c20[{i},{j}]")
# '1-21'
for i in N:
    for j in N:
        model.addConstr(x[i, j] + y[i, j] <= 1, name=f"c21[{i},{j}]")
# '1-22'
for i in N:
    for h in N:
        for j in N:
            model.addConstr(ns1[i, j] + KM * (2 - x[i, h] - x[h, j]) >= 0 + ke, name=f"c22[{i},{h},{j}]")
# '1-23'
for i in N:
    for h in N:
        for j in N:
            model.addConstr(ns1[i, j] + KM * (2 - y[i, h] - y[h, j]) >= 0, name=f"c23[{i},{h},{j}]")
# '1-24'
for i in N:
    for h in N:
        for j in N:
            model.addConstr(ns1[i, j] - KM * (2 - y[i, h] - y[h, j]) <= 0, name=f"c24[{i},{h},{j}]")
# '1-25'
for i in N:
    for h in N:
        for j in N:
            model.addConstr(ns1[i, j] + KM * (2 - x[i, h] - y[h, j]) >= 0 + ke, name=f"c25[{i},{h},{j}]")
# '1-26'
for i in N:
    for h in N:
        for j in N:
            model.addConstr(ns1[i, j] + KM * (2 - y[i, h] - x[h, j]) >= 0 + ke, name=f"c26[{i},{h},{j}]")
# '1-27'
for i in N:
    for h in N:
        for j in N:
            model.addConstr(ns1[i, h] + ns1[h, j] - ns1[i, j] <= f[i, h, j], name=f"c27[{i},{h},{j}]")
# '1-28'
for i in N:
    for h in N:
        for j in N:
            model.addConstr(-ns1[i, h] - ns1[h, j] + ns1[i, j] <= f[i, h, j], name=f"c28[{i},{h},{j}]")
# '1-29'
model.addConstr(sum(f[i, h, j] for i in N for h in N for j in N if i < h < j) <= (1 - aci2) *
                ((tau * n * (n-1) * (n-2)) / 2), name="c29")
# '1-30' 已经在定义变量的时候进行了约束
# '1-31'
for i in N:
    for j in N:
        model.addConstr(ns1[i, j] == sum(((t - tau) * beta1[i, j, t]) for t in T), name=f"c31[{i},{j}]")
# '1-32' 已经在定义变量的时候进行了约束

# 1.8 编译具体模型
model.write('Model2.lp')

# 1.9 求解
model.optimize()
x_matrix = []
if model.Status == gp.GRB.OPTIMAL:
    for i in N:
        for j in N:
            for t in T:
                x_matrix.append(beta1[i, j, t].X)
                print(f"[{i},{j},{t}]:", beta1[i, j, t].X)
modified_p = np.reshape(np.array(x_matrix), (n, n, (2 * tau) + 1))
print("决策者的满足一致性要求的修正偏好矩阵:\n", modified_p)