import datetime
import json
import numpy as np
import pandas as pd
import re
from covid_data.disease_process import get_confirmed_cases, get_death
from covid_data.demographics_process import get_density, get_population
from covid_data.sentiment_process import get_sentiment
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import os

with open(os.path.dirname(__file__)+"/processed_data/node_to_id.csv", 'r', encoding='UTF-8') as f:
    node_to_id = json.load(f)

node = pd.read_csv(os.path.dirname(__file__)+"/processed_data/node_0325.csv", dtype = {"0":np.float64, "1":np.float64,"2":np.float64,"3":np.float64,"4":np.float64})
def get_subGraph(type,name):
    sub_Graph_name_id = {}
    sub_Graph_id_name = {}
    index_id = {}
    node_name_list = []
    i = 0
    for node, id in node_to_id.items():
        node_name = list(filter(None, node.split('|')))
        if (len(node_name) > type+1 or len(node_name) == type+1):
            if (node_name[type] == name):
                # print(len(node_name))
                sub_Graph_name_id[node] = id
                sub_Graph_id_name[id] = node
                node_name_list.append(id)
                index_id[node] = i
                i = i + 1

    node = pd.read_csv("../covid_data/processed_data/node_0410.csv",
                       dtype={"0": np.float64, "1": np.float64, "2": np.float64, "3": np.float64, "4": np.float64})
    edge = pd.read_csv('../res/graph.csv')
    edge.drop(columns=['else'], inplace=True, axis=1)
    edge_1 = edge[edge["edge_from"].isin(list(sub_Graph_id_name.values()))]
    edge_2 = edge_1[edge_1["edge_to"].isin(list(sub_Graph_id_name.values()))]
    edge_near = edge_2[edge_2["edge_type"] == 1]
    edge_append = pd.DataFrame()
    edge_append["edge_from"] = edge_near["edge_to"]
    edge_append["edge_to"] = edge_near["edge_from"]
    edge_append["edge_type"] = edge_near["edge_type"]
    sub_graph_edge = edge_2.append(edge_append)
    node_from = sub_graph_edge['edge_from'].map(lambda x: index_id[x]).to_numpy()
    node_to = sub_graph_edge['edge_to'].map(lambda x: index_id[x]).to_numpy()
    node_attr = node.iloc[node_name_list, 1:].to_numpy()  # 节点属性
    edge_attr = sub_graph_edge["edge_type"].to_numpy()  # 边的属性
    edge_index = np.asarray([node_from, node_to])  # 边的索引
    node_attr = torch.from_numpy(node_attr).double()
    edge_index = torch.from_numpy(edge_index).long()
    edge_attr = torch.from_numpy(edge_attr)
    # node_attr =torch.rand(1142,5).double()
    return Data(x=node_attr, edge_index = edge_index, edge_attr = edge_attr)

def draw(Data):
    G = to_networkx(Data)
    nx.draw(G)
    plt.savefig("path.png")
    plt.show()

if __name__ == '__main__':
     data = get_subGraph(2, "Washington")
     draw(data)







