import re
state_name = {}
name_state = {}
for line in open("../res/USA_state.txt"):
    m =re.match('[\u4e00-\u9fa5]+\s([A-Za-z|\s]+)\s缩写：([A-Za-z]+)\s',line)
    # print(line)
    state_name[m.group(2)] = m.group(1)
    name_state[m.group(1)] = m.group(2)


def get_state_name(name):
    return state_name[name]


def name_to_state(state):#将州的全称转换为简称
    return name_state[state]

