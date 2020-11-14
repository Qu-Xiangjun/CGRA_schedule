from pulp import *
import pandas as pd
import numpy as np
import os


def get_data():
    """
    获取输入图数据
    @return x : n行，每一行数据的格式为 
                节点编号  子节点1  边类型  子节点2  边类型.1  
                子节点3  边类型.2  子节点4  边类型.3  
                最早时间步  最晚时间步  节点类型  有无父节点
    """
    # 读取数据
    filename = os.getcwd()+'\\example1.xls'
    data = pd.read_excel(filename)
    print("input data：")
    print(data)
    x = np.array([list(data[u'节点编号']), list(data[u'子节点1']), list(data[u'边类型1']),
                  list(data[u'子节点2']), list(data[u'边类型2']),
                  list(data[u'子节点3']), list(data[u'边类型3']),
                  list(data[u'子节点4']), list(data[u'边类型4']),
                  list(data[u'最早时间步']), list(data[u'最晚时间步']),
                  list(data[u'节点类型']), list(data[u'有无父节点'])])
    x = x.T
    x = x.tolist()
    print(x)
    print()

    return x


def output_file(data):
    # 任意的多组列表
    data = np.array(data)
    data = data.T
    data.tolist()

    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'节点编号': data[0], '时间步': data[1], '子节点1': data[2], '子节点2': data[3],
                              '子节点3': data[4], '子节点4': data[5], '节点类型': data[6], '原节点编号': data[7]})

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(os.getcwd()+"\\result.csv", index=False, sep=',')


class opterator:
    """
    一个算子类
    """

    def __init__(self):
        self.id = 0  # 算子编号
        self.generation = False  # 是否迭代有子节点
        self.earlier_time_step = 0  # 最早时间步
        self.lastest_time_step = 0  # 最晚时间步
        self.router_lastest_time_step = 0  # 可以被插入的路由节点最晚时间步
        self.children = []  # [子节点编号+边类型] 边类型：0为内依赖，1为迭代间依赖
        self.moves = False  # 是否有机动性，0为否，1为有，即lastest_time_step>earlier_time_step
        self.moves_router = []  # 算子的路由节点可放置范围 [earlier_time_step，router_lastest_time_step]
        self.start_op = False  # 是否为起始节点


def scheduled(data, npe):
    """
    调度函数
    @x 输入列表，每个字列表为一个算子的数据
    @npe PEA中的PE数量
    """
    key_route_length = 0  # 关键路径长度  即时间步从0到key_route_length-1
    op_list = []  # 算子列表
    E = []  # 依赖边集合 [[父节点，子节点,边类型(1为迭代边)]]
    Router = []  # 可路由算子列表

    # 初始化数据
    for x in data:
        op = opterator()
        op.id = int(x[0])
        for i in range(1, 8, 2):
            if(x[i] != 0):
                op.children.append( [int(x[i]), int(x[i+1])] )
                E.append( [int(x[0]), int(x[i]), int(x[i+1])] )
        op.children = [[x[1], x[2]], [x[3], x[4]], [x[5], x[6]], [x[7], x[8]]]
        op.earlier_time_step = int(x[9])
        op.lastest_time_step = int(x[10])
        if(int(x[11]) == 1):
            op.moves = True

        if(int(x[12]) == 0):  # 无父节点
            op.start_op = True
        # 计算 可以被插入的路由节点最晚时间步
        # 在找到kernel后计算

        for i in op.children:
            if(i[1] == 1):
                op.generation = True
        if(op.lastest_time_step > key_route_length):  # 找关键路径的深度
            key_route_length = op.lastest_time_step

        op_list.append(op)

    # 找出kernel
    # 满足基本的启动间隔约束
    # 满足在无路由节点下的pe数量约束 ( 有路由节点的PE数量约束放在ILP中进行 )
    for ii in range(2, key_route_length+1):  # 遍历得到最小符合的ii，由于1不可能实现，从2开始
        flag = True  # 符合flag
        # 启动间隔约束
        for edge in E:
            if(edge[2] == 0):
                continue
            else:  # 找到启动边
                t1 = data[edge[0]-1][9]  # 父节点的最早时间步
                t2 = data[edge[1]-1][9] + ii  # 子节点在下一迭代中的最早时间步
                if(t1 < t2):
                    flag = True
                else:
                    flag = False
                    break
        if(flag == False):
            continue
        # PE 数量约束
        count = [0 for i in range(ii)]  # 每一行的pe使用数
        for op in op_list:
            count[op.earlier_time_step % ii] += 1
        for cnt in count:
            if(cnt > npe):
                flag = False
        if(flag == False):
            continue

    # 在已知ii的基础上进行路由算子集合与路由节点最晚时间步计算
    for x in data:
        op = op_list[x[0] - 1 ]
        max_temp = 0
        for child in op.children:
            t2 = 0
            if(child[1] == 1): # 可迭代算子，要加上ii
                t2 = data[child[0]-1][10] + ii # 最晚时间步加ii
            else:
                t2 = data[child[0]-1][10]
            if(max_temp < t2-1):
                max_temp = t2-1
        op.router_lastest_time_step = max_temp
        op.moves_router = [op.earlier_time_step,op.router_lastest_time_step]  # 算子的路由节点可放置范围 
        if(max_temp > 0): # 可路由
            Router.append(op)
    
    # 添加约束
    prob = LpProblem("CGRA",LpMinimize) # 最小化问题
    x_var_list = [] # 每个算子x的变量集合，每个字列表
    for i in range(len(data)):
        op = op_list[i]
        tmp = data[i-1]
        ls1 = []
        for j in range(op.moves_router[0],op.lastest_time_step+1):
            ls2 = []
            for k in range(j,op.moves_router[1]+1):
                var = pulp.LpVariable( "x_{}_{}_{}".format(i+1,j,k),lowBound=0,cat='Integer')
                ls2.append(var)
            ls1.append(ls2)
        print(ls1)
if __name__ == '__main__':
    scheduled(get_data(),16)
    
