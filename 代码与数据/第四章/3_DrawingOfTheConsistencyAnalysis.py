# -*- codeing = utf-8 -*-
# @Time : 2024-02-19 21:27
# @Author : 吴国林
# @File : 3_DrawingOfTheConsistencyAnalysis.py
# @Software : PyCharm
import numpy as np
from matplotlib import pyplot as plt
# from matplotlib.font_manager import FontProperties # 导入FontProperties
# font = FontProperties(fname="SimHei.ttf", size=14)
import matplotlib as mpl

mpl.rcParams["font.family"] = "FangSong"  # 设置字体
mpl.rcParams["axes.unicode_minus"] = False  # 正常显示负号

n = 4
tau = 3

data_1 = np.loadtxt('Original_20_DLPRs_aci.txt')
data_2_0 = np.loadtxt('ObjectiveValue_20_DLPRs.txt')
data_2 = data_2_0 / ((n * (n - 1) / 2) * ((2 * tau) + 1))
data_3 = np.loadtxt('Consistent_20_DLPRs_aci.txt')
print(data_2)
y_1 = np.reshape(data_1, (20, 1))
y_2 = np.reshape(data_2, (20, 1))
y_3 = np.reshape(data_3, (20, 1))

x = range(0, 20, 1)
x_labels = ["e_1", "e_2", "e_3", "e_4", "e_5", "e_6", "e_7", "e_8", "e_9", "e_10",
            "e_11", "e_12", "e_13", "e_14", "e_15", "e_16", "e_17", "e_18", "e_19", "e_20"]
fig = plt.figure(figsize=(12, 5), dpi=90)
plt.plot(x, y_1, label="原始的ACI", color="#ff3399", linestyle=':', linewidth='3')
plt.plot(x, y_2, label="调整的幅度", color="#ff3300", linestyle='-.', linewidth='3')
plt.plot(x, y_3, label="调整后的ACI", color="Cyan", linestyle='--', linewidth='3')
plt.xticks(list(x)[::1], x_labels[::1], fontsize=25)
y_ticks = [i / 10 for i in range(0, 11, 1)]
plt.yticks(y_ticks, fontsize=25)
plt.xlabel('决策者', size=30)
plt.ylabel('数值标度', size=30)
plt.title("使用模型M-2调整前后的相关指标的变化情况", size=30)

plt.grid(alpha=0.25)  # alpha 透明度，当alpha=1表示完全不透明

plt.legend(loc="center right", fontsize=20)

plt.show()
