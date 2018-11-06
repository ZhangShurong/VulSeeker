import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools

adj_matrix_all_1=[]
element=[[(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (4, 7)],\
         [(1, 5), (2, 3), (3, 4), (4, 4), (5, 2)],\
         [(1, 2), (2, 5), (3, 4), (4, 5), (4, 6), (5, 1)]]
for i in range(2):
    G = nx.Graph(
        element[i])
    # G.add_edge(0,1,weight=2)
    # G.add_edge(1,0)
    # G.add_edge(2,2,weight=3)
    # G.add_edge(2,2)
    nx.draw_networkx(G)
    plt.show()
    adj_arr = np.array(nx.to_numpy_matrix(G))
    adj_str = adj_arr.astype(np.string_)
    adj_matrix_all_1.append(",".join(str(list(itertools.chain.from_iterable(adj_str)))))
    print(adj_matrix_all_1[i])
    print(adj_arr)
    # matrix([[ 0.,  2.,  0.],
    #         [ 1.,  0.,  0.],
    #         [ 0.,  0.,  4.]])