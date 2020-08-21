import pandas as pd
import datetime
from covid_data.usa_state import name_to_state

confirmed_cases = pd.read_excel('../res/confirmed_case.xlsx', sheet_name='Sheet1')
death = pd.read_excel('../res/death.xlsx', sheet_name='Sheet1')
# state = confirmed_cases[confirmed_cases.state=='AK']
# country = state[state.country == 'Anchorage']

# print(country[confirmed_cases(2020,4,10)])



def get_confirmed_cases(node_type,node_name,time):#获取某个节点的确诊人数 输入为节点类型，节点名称，查询时间
    if node_type == 0:
        return  confirmed_cases[time].sum()

    elif node_type == 1:
        state_confirm = confirmed_cases[confirmed_cases.state == name_to_state(node_name[1])][time].sum()
        return state_confirm
    else:
        state = confirmed_cases[confirmed_cases.state == name_to_state(node_name[1])]
        country = state[state.country == node_name[node_type]]
        if (country.empty):
            return 0
        else:
            return country[time].sum()


def get_death(node_type,node_name,time):#获取某个节点的确诊人数 输入为节点类型，节点名称，查询时间
    if node_type == 0:
        return  death[time].sum()

    elif node_type == 1:
        state_confirm = death[death.state == name_to_state(node_name[1])][time].sum()
        return state_confirm
    else:
        state = death[death.state == name_to_state(node_name[1])]
        country = state[state.county == node_name[node_type]]
        if (country.empty):
            return 0
        else:
            return country[time].sum()


# print(get_confirmed_cases(3, ['USA', 'Alabama', 'Chambers', 'Abanda'],datetime.datetime(2020, 4, 10)))