# -*- codeing = utf-8 -*-
# @Time : 2024-02-10 17:21
# @Author : 吴国林
# @File : 1_TheModelForConsistencyOfDlprs.py
# @Software : PyCharm
import numpy as np
import gurobipy as gp

# 1.1 已知量与数据
data = np.loadtxt('All_Subgroups_DLPRs.txt')
m = 8    # 决策者的数量
n = 4    # 待选择或排序的备选方案的数量
tau = 3  # LTS的表示 2 * tau + 1 = 7
M = list(range(m))  # 生成表示各个决策者的索引列表
N = list(range(n))  # 生成表示各个备选方案的索引列表
T = list(range(2 * tau + 1))    # 生成表示各个语言术语的索引列表 T = [-3, -2, -1, 0, 1, 2, 3] --> [0, 1, 2, 3, 4, 5, 6]
T1 = list(range(2 * tau))   # 剔除 tau


# 1.2 查看原始偏好数据
B = np.reshape(data, (m, n, n, (2 * tau) + 1))    # B[k,i,j,t] 包含 20个 DLPRs

# 设置模型参数的初始值
aci2 = 0.8   # 根据特定决策问题预先给定的想要达成的个体的加型基数一致性水平
KM = 10     # 一个较大的正数，用于0-1整数规划建模
ke = 0.001  # 一个较小的正数，用于将严格不等式转化为基本不等式
# z1_list = [4, 4, 6, 4, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5]  # B中各个DLPRs的最低犹豫度
# z2_list = [7, 6, 7, 7, 7, 7, 6, 5, 6, 7, 7, 5, 6, 6, 7, 6, 5, 5, 5, 6]  # B中各个DLPRs的最大犹豫度

z1_list = [1, 1, 1, 1, 1, 1, 1, 1]  # B中各个DLPRs的最低犹豫度
z2_list = [7, 7, 7, 7, 7, 7, 7, 7]  # B中各个DLPRs的最大犹豫度

def model2consistency(beta, z1, z2):
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
    x_list = []
    if model.Status == gp.GRB.OPTIMAL:
        for i in N:
            for j in N:
                for t in T:
                    x_list.append(round(beta1[i, j, t].X, 2))
                    # print(f"[{i},{j},{t}]:",beta1[i, j, t].X)
    return x_list, model.getObjective().getValue()


if __name__ == '__main__':
    modified_B = []
    obj_values = []
    for k in range(m):
        beta_k = B[k, :, :]
        z1_k = z1_list[k]
        z2_k = z2_list[k]
        print(f"------------------------对第{k + 1}个决策者的语言分布偏好关系的和理性分析过程------------------------")
        modified_b, obj_res = model2consistency(beta=beta_k, z1=z1_k, z2=z2_k)
        # modified_B.append(modified_b)
        obj_values.append(obj_res)
        print(f"决策者{k + 1}的满足一致性要求的修正偏好矩阵:\n", np.reshape(modified_b, (n, n, 2 * tau + 1)))
        for num in modified_b:
            modified_B.append(num)
    print(f"总列表：\n", modified_B)

    # with open('Consistent_20_DLPRs.txt', 'w') as file:
    #     for k in range(0, len(modified_B), (2 * tau) + 1):
    #         file.write(str(modified_B[k:(k + (2 * tau + 1))]) + '\n')  # 将数组元素逐行写入txt 文件
    # file.close()

    with open('Consistent_20_DLPRs.txt', 'w') as file:
        for k in modified_B:
            file.write(str(k) + '\n')  # 将数组元素逐行写入txt 文件
    # 关闭文件对象
    file.close()

    print(f"目标函数值：", obj_values)
    with open('ObjectiveValue_20_DLPRs.txt', 'w') as file:
        for item in obj_values:
            file.write(str(item) + '\n')  # 将数组元素逐行写入txt 文件
    # 关闭文件对象
    file.close()