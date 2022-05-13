import networkx as nx
import numpy as np

def create_graph_edgelist(graph_file,num_nodes):
    G = nx.DiGraph()
    f = open(graph_file)
    line = f.readline()
    data_list = []
    while line:
        num = list(map(float, line.split()))
        data_list.append(num)
        line = f.readline()
    f.close()
    data_array = np.array(data_list)
    for i in range(len(data_array)):                    #添加边
        G.add_edge(int(data_array[i][0]), int(data_array[i][1]))
        G.get_edge_data(data_array[i][0], data_array[i][1])['weight'] = data_array[i][2]

    for i in range(1, int(num_nodes)):                   #添加孤立节点
        if i not in G.nodes():
            G.add_node(i)
    return G

# G = create_graph_edgelist('data/test_graph.txt',34)
#
# edges_raw = G.edges(data=True)
# print(edges_raw)
# edge_distribution = np.array([attr['weight'] for _, _, attr in edges_raw], dtype=np.float32)
# print(edge_distribution)
#
# #for edge in G.edges:
# print(G.nodes)
# print(G.edges)
# print(G.get_edge_data(1,32))