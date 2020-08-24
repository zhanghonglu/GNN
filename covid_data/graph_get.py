import datetime
import json
import numpy as np
import pandas as pd
import re
from covid_data.disease_process import get_confirmed_cases, get_death
from covid_data.demographics_process import get_density, get_population
from covid_data.sentiment_process import get_sentiment
graph_df =pd.read_csv('../res/graph.csv')
#   得到节点名称与索引的映射关系并写入json文件
# node_set = list(set.union(set(graph_df['edge_from'].array),set(graph_df['edge_to'].array)))
# node_to_id={ node:i for i,node in enumerate(node_set)}
# json_node_id = json.dumps(node_to_id, indent=4)
# with open('./processed_data/node_to_id.csv', 'w') as json_file:
#     json_file.write(json_node_id)
with open("./processed_data/node_to_id.csv", 'r', encoding='UTF-8') as f:
    node_to_id = json.load(f)
# node_from =graph_df['edge_from'].map(lambda x:node_to_id[x])
# node_to =graph_df['edge_to'].map(lambda x:node_to_id[x])
# edge_type = graph_df['edge_type'].apply(lambda x:int(x))
#图的节点类属性 [类型，确诊人数，新增人数，死亡人数，人口，人口密度，舆论认知]
                #类型信息  0：USA  1:state 2country  3 city



# edge_pd = pd.DataFrame({"node_from":node_from,"node_to":node_to})
# edge_pd.to_csv('./processed_data/edge_01.csv')


# def get_edge_index():
#     return [node_from,node_to]


def get_data_x(time):
    data_x = []
    for node, id in node_to_id.items():
        print("{num}/{totoal}".format(num = id, totoal =len(node_to_id)))
        node_name = list(filter(None,node.split('|')))
        node_type = len(node_name)-1
        # print("节点名称:{node_name} 节点类型:{node_type}".format(node_name = node_name, node_type =node_type),end="|")
        confirmed = get_confirmed_cases(node_type, node_name, time)
        # print("确诊人数：{confirm}".format(confirm = confirmed),end="|")
        death = get_death(node_type, node_name, time)
        # print("死亡人数：{death}".format(death=death), end="|")
        population = get_population(node_type, node_name)
        # print("总人数：{population}".format(population=population), end="|")
        density = get_density(node_type, node_name)
        # print("人口密度：{density}".format(density=density), end="|")
        sentiment = get_sentiment(node_type, node_name)
        # print("舆论：{sentiment}".format(sentiment=sentiment))
        x =[confirmed, death, population, density, sentiment]
        print(x)
        data_x.append(x)
    return data_x

# print(get_data_x(datetime.datetime(2020, 4, 10)))
# edge = get_edge_index()
x_pd = pd.DataFrame(get_data_x(datetime.datetime(2020, 4, 25)))
x_pd.to_csv("./processed_data/node_0410.csv")

# def get_edge_attr():
#
#     return ( [ [index, value]for index,value in enumerate(edge_type.array)])
#
#
# pass


