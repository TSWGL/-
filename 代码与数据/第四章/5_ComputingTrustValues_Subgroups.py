# -*- codeing = utf-8 -*-
# @Time : 2024-02-20 15:12
# @Author : 吴国林
# @File : 5_ComputingTrustValues_Subgroups.py
# @Software : PyCharm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle


data = np.loadtxt('Social_trust_matrix_2.txt')
m = 20
tau_1 = 1
T_1 = list(range((2 * tau_1) + 1))
TD = np.reshape(data, (m, m, (2 * tau_1) + 1))

C = {'C_1': [0],
     'C_2': [1],
     'C_3': [2],
     'C_4': [3],
     'C_5': [4],
     'C_6': [5],
     'C_7': [6, 13, 14, 15],
     'C_8': [7, 8, 9, 10, 11, 12, 16, 17, 18, 19]
     }


td = {}
for cp in C.keys():
    for cq in C.keys():
        if cq != cp:
            td[cp, cq] = (1 / (len(C[cp]) * len(C[cq]))) * sum(
                    ((sum((t - tau_1) * TD[k, h, t] for t in T_1) + tau_1) / (2 * tau_1))
                    for k in C[cp]
                    for h in C[cq]
            )

print(td)
temp_value = []
for key in td.keys():
    temp_value.append(td[key])
print(f"temp_value:\n", temp_value)
temp_dic = []
for i in range(8):
    for j in range(8):
        if i != j:
            temp_dic.append((i, j))
print(f"temp_dic:\n", temp_dic)
td_save = {}
for k in range(len(temp_dic)):
    td_save[temp_dic[k]] = temp_value[k]
print(f"保存：\n", td_save)
with open("All_Subgroups_social_trust.pkl", 'wb') as f:
    pickle.dump(td_save, f, pickle.HIGHEST_PROTOCOL)


G = nx.DiGraph()
# G = nx.MultiDiGraph()

nodes = ['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7', 'C_8']
# for node1 in nodes:
#     for node2 in nodes:
#         if node2 != node1:
#             edges.append((node1, node2))
G.add_nodes_from(nodes)
for key in td.keys():
    G.add_edge(key[0], key[1], weight=round(td[key], 2))
nx.draw(G, pos=nx.shell_layout(G), with_labels=True, connectionstyle='arc3, rad = 0.2', node_color='g', edge_color='b',
        node_size=400, width=0.6, font_size=10, alpha=0.5)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos=nx.shell_layout(G), edge_labels=labels, label_pos=0.16,
                             verticalalignment='top', horizontalalignment='right',
                             font_size=8, alpha=1)


# start = []
# to = []
# value = []
# for key in td.keys():
#     start.append(key[0])
#     to.append(key[1])
#     value.append(td[key])
# for j in range(0, len(start)):
#     G.add_weighted_edges_from([(start[j], to[j], value[j])])  # 边的起点，终点，权重
# nx.draw(G, with_labels=True,pos=nx.circular_layout(G), connectionstyle='arc3, rad = 0.2', node_color='g', edge_color='b',
#         width=[float(v['weight']) for (r, c, v) in G.edges(data=True)], node_size=400, font_size=10, alpha=0.5)

plt.show()
