import pandas as pd


demographics = pd.read_excel('../res/demographics.xlsx', sheet_name='Sheet1')


def get_population(node_type,node_name):
    if node_type == 0:
        return demographics['population'].sum()
    elif node_type == 1:
        return demographics[demographics.state_name == node_name[node_type]]["population"].sum()
    elif node_type == 2:
        return demographics[(demographics.county_name == node_name[2])&(demographics.state_name == node_name[1])]["population"].sum()
    elif node_type == 3:
        return demographics[(demographics.city == node_name[3]) & (demographics.county_name == node_name[2]) & (demographics.state_name == node_name[1])]["population"].sum()


def get_density(node_type,node_name):
    if node_type == 3:
        return demographics[(demographics.city == node_name[3]) & (demographics.county_name == node_name[2]) & (demographics.state_name == node_name[1])]["density"].sum()
    elif node_type == 2:
        country = demographics[(demographics.state_name == node_name[1]) & (demographics.county_name == node_name[2])]
        return (country["population"].sum())/(country["area"].sum())
    elif node_type == 1:
        state = demographics[demographics.state_name == node_name[1]]
        return state["population"].sum()/state["area"].sum()
    elif node_type == 0:
        return  demographics["population"].sum()/demographics["area"].sum()




