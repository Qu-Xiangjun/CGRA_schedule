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
    filename = 'example2.xls'
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
    """
    输出表到excel文件
    @return x : n行，每一行数据的格式为
                节点编号  子节点1  边类型  子节点2  边类型.1
                子节点3  边类型.2  子节点4  边类型.3
    """
    # 任意的多组列表
    # data = np.array(data)
    data = data.T
    data.tolist()

    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'节点编号': data[0], '时间步': data[1], '子节点1': data[2], '子节点2': data[3],
                              '子节点3': data[4], '子节点4': data[5], '节点类型': data[6], '原节点编号': data[7]})

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(os.getcwd() + "\\result2.csv", index=False, sep=',')


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

class output_op:
    """
    输出算子类
    """
    def __init__(self):
        self.id = 0
        self.old_id = 0
        self.timestamp = 0
        self.children = []
        self.specie = 0 # 0为算子节点 1 为插入的pe路由节点 2 为插入的store节点 3位插入的load节点


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
        op.id = int(x[0])  # 编号
        for i in range(1, 8, 2):  # 子节点添加
            if (x[i] != 0):  # 不为空的子节点
                op.children.append([int(x[i]), int(x[i + 1])])
                E.append([int(x[0]), int(x[i]), int(x[i + 1])])
        # op.children = [[x[1], x[2]], [x[3], x[4]], [x[5], x[6]], [x[7], x[8]]]
        op.earlier_time_step = int(x[9])  # 最早时间步
        op.lastest_time_step = int(x[10])  # 最晚时间步
        if (int(x[11]) == 1):  # 机动性节点
            op.moves = True

        if (int(x[12]) == 0):  # 无父节点
            op.start_op = True

        # 计算 可以被插入的路由节点最晚时间步
        # 在找到kernel后计算

        for i in op.children:
            if (i[1] == 1):  # 有迭代子节点
                op.generation = True
        if (op.lastest_time_step + 1 > key_route_length):  # 找关键路径的深度
            key_route_length = op.lastest_time_step + 1  # 关键路径（1~n) = 最大时间步

        op_list.append(op)

    # 找出kernel
    # 满足基本的启动间隔约束
    # 满足在无路由节点下的pe数量约束 ( 有路由节点的PE数量约束放在ILP中进行 )
    ii = 2
    for ii in range(1, key_route_length + 1):  # 遍历得到最小符合的ii，由于1不可能实现，从2开始
        flag = True  # 符合flag

        # 启动间隔约束
        for edge in E:
            if (edge[2] == 0):
                continue

            else:  # 找到启动边
                t1 = data[edge[0] - 1][9]  # 父节点的最早时间步
                t2 = data[edge[1] - 1][9] + ii  # 子节点在下一迭代中的最早时间步
                if (t1 < t2):
                    flag = True
                else:
                    flag = False
                    break
        if (flag == False):
            continue

        # PE 数量约束
        count = [0 for i in range(ii)]  # 初始化每一行的pe使用数0
        for op in op_list:
            count[op.earlier_time_step % ii] += 1
        for cnt in count:
            if (cnt > npe):
                flag = False
        if (flag == False):
            continue

        # 长依赖约束
        # 检查所有边依赖的 ti %ii == tj%ii
        for edge in E:
            ti = data[edge[0] - 1][9]
            tj = data[edge[1] - 1][9]
            if (ti % ii == tj % ii):
                flag = False
                break
        if (flag == False):
            continue
        else:
            break

    for II in range(ii, key_route_length + 1):  # 若无解，则循环
        # 在已知ii的基础上进行路由算子集合与路由节点最晚时间步计算
        for x in data:
            op = op_list[x[0] - 1]  # 取出路由算子
            max_temp = 0  # 路由最晚时间步
            for child in op.children:
                t2 = 0
                if (child[1] == 1):  # 可迭代算子，要加上ii
                    t2 = data[child[0] - 1][10] + II  # 最晚时间步加ii
                else:
                    t2 = data[child[0] - 1][10]
                if (max_temp < t2 - 1):  # 最晚路由算子为子节点的最晚时间-1
                    max_temp = t2 - 1
            if (len(op.children) == 0):  # 没有子节点的末尾节点，设置路由范围为相同的最早开始步
                max_temp = op.earlier_time_step
            op.router_lastest_time_step = max_temp
            op.moves_router = [op.earlier_time_step,
                               op.router_lastest_time_step]  # 算子的路由节点可放置范围
            if (max_temp > 0):  # 可路由
                Router.append(op)

        # 求解
        # 添加自变量
        prob = LpProblem("CGRA", LpMinimize)  # 最小化问题
        # 添加Npe变量,目标方程中约束
        Npe = pulp.LpVariable("Npe", lowBound=0, cat='Integer')

        x_var_list = []  # 每个算子x的变量集合，每个字列表
        for i in range(len(data)):
            op = op_list[i]
            tmp = data[i]
            ls1 = []  # 每个算子的所有节点集合的列表
            # 算子的机动范围
            for j in range(op.moves_router[0], op.lastest_time_step + 1):
                ls2 = []  # 每个算子的节点列表
                # 算子的可路由范围
                for k in range(j, op.moves_router[1] + 1):
                    var = pulp.LpVariable("x_{}_{}_{}".format(
                        i + 1, j, k), lowBound=0, cat='Integer')
                    ls2.append(var)
                ls1.append(ls2)
            print(ls1)
            x_var_list.append(ls1)
        print()

        '''
        lql part: memory constraint and variables
        '''
        y_var_list = []
        z_var_list = []
        tm = 2  # xkl说这样写比较规范
        for i in range(len(data)):
            op = op_list[i]
            ls1 = []  # 每个算子的所有节点集合的列表
            # 算子的机动范围
            sn, ln = op.earlier_time_step, op.lastest_time_step
            for j in range(sn, ln + 1):
                ls2 = []  # 每个算子的节点列表
                # 算子的可路由范围
                j_end = op.router_lastest_time_step - tm + 1
                for k in range(j+1, j_end + 1):
                    var = pulp.LpVariable("y_{}_{}_{}".format(
                        i + 1, j, k), lowBound=0, cat='Integer')
                    ls2.append(var)
                ls1.append(ls2)
            print(ls1)
            y_var_list.append(ls1)

        for i in range(len(data)):
            op = op_list[i]
            ls1 = []  # 每个算子的所有节点集合的列表
            # 算子的机动范围
            sn, ln = op.earlier_time_step, op.lastest_time_step
            for j in range(sn, ln + 1):
                ls2 = []  # 每个算子的节点列表
                # 算子的可路由范围
                j_start = j + tm
                j_end = op.router_lastest_time_step
                for k in range(j_start, j_end + 1):
                    var = pulp.LpVariable("z_{}_{}_{}".format(
                        i + 1, j, k), lowBound=0, cat='Integer')
                    ls2.append(var)
                ls1.append(ls2)
            print(ls1)
            z_var_list.append(ls1)

        # 添加约束
        constraints = []  # 约束列表

        # 唯一性约束
        print("唯一性")
        for i in range(len(x_var_list)):
            constraints.append(
                lpSum(x_var_list[i][j][0]
                      for j in range(len(x_var_list[i]))) == 1
            )
            print(lpSum(x_var_list[i][j][0]
                        for j in range(len(x_var_list[i]))) == 1)

        # 排他性
        print("排他性")
        for i in range(len(x_var_list)):
            temp_var_list = x_var_list[i]
            j = len(temp_var_list)  # 变量的组数
            k = len(temp_var_list[0])  # 最长的变量组的变量数
            for j1 in range(j):  # 当前调度算子的组
                for j2 in range(j):  # 非调度算子的组
                    if (j1 != j2):  # 判断j2不是调度算子的组号
                        for m in range(1, len(temp_var_list[j2])):
                            # 遍历非当前调度算子组的路由算子，即从下标1开始
                            # 每一非选中调度算子组的其他组路由算子加选中的调度算子都小于等于1
                            constraints.append(
                                temp_var_list[j1][0] +
                                temp_var_list[j2][m] <= 1
                            )
                            print(
                                temp_var_list[j1][0] +
                                temp_var_list[j2][m] <= 1
                            )
        '''
        lql part:  加入y，z相关的排他性约束
        '''
        print("排他性约束_memory：")
        for n in range(len(data)):
            x_tmp_list = x_var_list[n]
            y_tmp_list = y_var_list[n]

            i1 = len(x_tmp_list)
            i2 = len(y_tmp_list)
            for ii1 in range(i1):
                for ii2 in range(i2):
                    if(ii1 != ii2):
                        for j in range(len(y_tmp_list[ii2])):
                            constraints.append(
                                x_tmp_list[ii1][0] + y_tmp_list[ii2][j] <= 1
                            )

                            print(
                                x_tmp_list[ii1][0] + y_tmp_list[ii2][j] <= 1
                            )

        for n in range(len(data)):
            x_tmp_list = x_var_list[n]
            z_tmp_list = z_var_list[n]
            # print("---test---")
            # print(x_tmp_list)
            # print(z_tmp_list)
            i1 = len(x_tmp_list)
            i2 = len(z_tmp_list)
            for ii1 in range(i1):
                for ii2 in range(i2):
                    if(ii1 != ii2):
                        for j in range(len(z_tmp_list[ii2])):
                            constraints.append(
                                x_tmp_list[ii1][0] + z_tmp_list[ii2][j] <= 1
                            )
                            print(
                                x_tmp_list[ii1][0] + z_tmp_list[ii2][j] <= 1
                            )

        '''
        插入load算子和store算子同时存在
        '''
        print("load store 约束")
        for n in range(len(data)):
            y_tmp_list = y_var_list[n]
            z_tmp_list = z_var_list[n]

            for i in range(len(y_tmp_list)):
                constraints.append(
                    lpSum(y_tmp_list[i][j1] for j1 in range(len(y_tmp_list[i]))) -
                    lpSum(z_tmp_list[i][j2]
                          for j2 in range(len(z_tmp_list[i]))) == 0
                )
                print(
                    lpSum(y_tmp_list[i][j1] for j1 in range(len(y_tmp_list[i]))) -
                    lpSum(z_tmp_list[i][j2]
                          for j2 in range(len(z_tmp_list[i]))) == 0
                )
                # for j1 in range(len(y_tmp_list[i])):
                #     for j2 in range(len(z_tmp_list[i])):
                #             constraints.append(
                #                 y_tmp_list[i][j1] - z_tmp_list[i][j2] == 0
                #             )
                #             print(y_tmp_list[i][j1] - z_tmp_list[i][j2] == 0)

        '''
        lql part :互斥
        '''
        print("互斥")
        for n in range(len(data)):
            y_tmp_list = y_var_list[n]
            x_tmp_list = x_var_list[n]
            for i in range(len(y_tmp_list)):
                for j1 in range(len(y_tmp_list[i])):
                    # 第0个是调度节点，非路由节点，排除
                    for j2 in range(1, len(x_tmp_list[i])):
                        constraints.append(
                            y_tmp_list[i][j1] + x_tmp_list[i][j2] <= 1
                        )
                        print(y_tmp_list[i][j1] + x_tmp_list[i][j2] <= 1)

        # 依赖约束
        # 父节点调度时间步小于子节点调度时间步
        print("依赖约束：父节点调度时间步小于子节点调度时间步")
        for edge in E:
            father = x_var_list[edge[0] - 1]  # 父变量集合
            kid = x_var_list[edge[1] - 1]  # 子变量集合
            if (edge[2] == 1):  # 迭代边
                for i in range(len(father)):  # 遍历每一个父的调度节点变量
                    constraints.append(
                        (i + data[edge[0] - 1][9]) * father[i][0] - lpSum((k +
                                                                           data[edge[1] - 1][9] + II) * kid[k][0] for k
                                                                          in range(len(kid))) <= -1
                    )
                    print(
                        (i + data[edge[0] - 1][9]) * father[i][0] - lpSum((k +
                                                                           data[edge[1] - 1][9] + II) * kid[k][0] for k
                                                                          in range(len(kid))) <= -1
                    )
            else:
                for i in range(len(father)):  # 遍历每一个父的调度节点变量
                    constraints.append(
                        (i + data[edge[0] - 1][9]) * father[i][0] - lpSum((k +
                                                                           data[edge[1] - 1][9]) * kid[k][0] for k in
                                                                          range(len(kid))) <= -1
                    )
                    print(
                        (i + data[edge[0] - 1][9]) * father[i][0] - lpSum((k +
                                                                           data[edge[1] - 1][9]) * kid[k][0] for k in
                                                                          range(len(kid))) <= -1
                    )

        # 父算子的所有路由节点都小于子节点中最大时间步
        print("依赖约束：父算子小于所有子节点中的最大时间步")
        for i in range(len(op_list)):
            max_sn = 0  # 子节点中最大时间步
            op = op_list[i]
            if (len(op.children) == 0):  # 无子节点，跳过约束
                continue
            for child in op.children:
                id = child[0]
                kid_op = op_list[id - 1]
                if (max_sn < kid_op.lastest_time_step):
                    max_sn = kid_op.lastest_time_step
            for j in range(len(x_var_list[i])):
                for k in range(len(x_var_list[i][j])):
                    temp = x_var_list[i][j]
                    constraints.append(
                        (k + op.earlier_time_step + j) * temp[k] - max_sn <= -1
                    )
                    print(
                        (k + op.earlier_time_step + j) * temp[k] - max_sn <= -1
                    )

        # PE资源约束
        print("PE约束")

        x_var = [[] for i in range(II)]  # 收集每个ii行积累的算子
        y_var = [[] for i in range(II)]
        z_var = [[] for i in range(II)]

        for i in range(len(x_var_list)):
            for j in range(len(x_var_list[i])):
                for k in range(len(x_var_list[i][j])):
                    x_var[(k + (data[i][9]) + j) %
                          II].append(x_var_list[i][j][k])
        for i in range(len(y_var_list)):
            for j in range(len(y_var_list[i])):
                for k in range(len(y_var_list[i][j])):
                    y_var[(k + (data[i][9]) + j) %
                          II].append(y_var_list[i][j][k])
        for i in range(len(z_var_list)):
            for j in range(len(z_var_list[i])):
                for k in range(len(z_var_list[i][j])):
                    z_var[(k + (data[i][9]) + j) %
                          II].append(z_var_list[i][j][k])

        # for i in range(len(x_var_list)):
        #     for j in range(len(x_var_list[i])):
        #         for k in range(len(x_var_list[i][j])):
        #             if i < len(y_var_list) and j < len(y_var_list[i]) and k < len(y_var_list[i][j]):
        #                 if i < len(z_var_list) and j < len(z_var_list[i]) and k < len(z_var_list[i][j]):
        #                     x_var[(k + (data[i][9]) + j) %
        #                           II].append(x_var_list[i][j][k])
        #                     y_var[(k + (data[i][9]) + j) %
        #                           II].append(y_var_list[i][j][k])
        #                     z_var[(k + (data[i][9]) + j) %
        #                           II].append(z_var_list[i][j][k])
        for tmp in x_var:
            print(tmp)
        for tmp in y_var:
            print(tmp)
        for tmp in z_var:
            print(tmp)

        for i in range(len(x_var)):
            constraints.append(
                lpSum(x_var[i]) + lpSum(y_var[i]) + lpSum(z_var[i]) <= Npe
            )
            print(lpSum(x_var[i]) + lpSum(y_var[i]) + lpSum(z_var[i]) <= Npe)
        # 约束Npe
        constraints.append(
            Npe <= npe
        )
        print(Npe <= npe)

        # 添加约束
        for item in constraints:
            prob += item

        # 目标方程
        print("目标方程")
        # 路由算子使用量
        Nins_ls = []
        for i in range(len(x_var_list)):
            for j in range(len(x_var_list[i])):
                for k in range(len(x_var_list[i][j])):
                    if (0 != k):
                        Nins_ls.append(x_var_list[i][j][k])
                        print(x_var_list[i][j][k])
        for i in range(len(y_var_list)):
            for j in range(len(y_var_list[i])):
                for k in range(len(y_var_list[i][j])):
                    if (0 != k):
                        Nins_ls.append(y_var_list[i][j][k])
                        print(y_var_list[i][j][k])
        for i in range(len(z_var_list)):
            for j in range(len(z_var_list[i])):
                for k in range(len(z_var_list[i][j])):
                    if (0 != k):
                        Nins_ls.append(z_var_list[i][j][k])
                        print(z_var_list[i][j][k])

        beta = len(Nins_ls)  # 目标方程系数项，等于路由算子最大使用量

        # for ls in x_var:
        #     prob += (lpSum(beta * ls[j])
        #              for j in range(len(ls))) - lpSum(Nins_ls)
        #     print(
        #         (lpSum(beta * ls[j]) for j in range(len(ls))) - lpSum(Nins_ls)
        #     )

        prob += (beta * Npe - lpSum(Nins_ls))
        print(beta * Npe - lpSum(Nins_ls))

        print()
        print("ii := ", ii)
        print()

        prob.solve()

        optimal_flag = True
        if ("Optimal" != LpStatus[prob.status]):  # 无最优解
            continue

        # 格式化输出表图
        Npe_ans = 0
        varValue_ans_ls = []
        for v in prob.variables():
            if(v.name == "Npe"):
                Npe_ans = v.varValue
            if(v.varValue == 1):
                varValue_ans_ls.append([v.name,v.varValue])
        ans_var_num = len(varValue_ans_ls) # 节点数量
        # 结果算子列表
        ans_op_ls = [[] for i in range(len(data))] # 每个id-1为下标存存储本原节点对应的所有pe
        r_num = len(data) # 调度节点数量
        # 初始化
        for i in varValue_ans_ls:
            tmp = i[0].split('_')
            var_class = tmp[0] # 变量类型 xyz
            var_id = int(tmp[1]) # id
            var_diaodu = int(tmp[2]) # 起始
            var_excute = int(tmp[3]) # 执行时间步
            out_op = output_op() # 输出算子对象
            if(var_diaodu == var_excute): # 调度节点
                out_op.old_id = out_op.id = var_id # 节点编号，原节点编号
                out_op.timestamp = var_excute # 执行时间步
                out_op.specie = 0 # 算子节点
                ans_op_ls[var_id-1].insert(0,out_op) # 调度节点插入第一位
            else: # 路由节点
                r_num += 1
                out_op.old_id = var_id
                out_op.id = r_num
                out_op.timestamp = var_excute # 执行时间步
                if(var_class == 'x'):
                    out_op.specie = 1 # 算子节点
                elif(var_class == 'y'):
                    out_op.specie = 2
                else:
                    out_op.specie = 3
                ans_op_ls[var_id-1].append(out_op)
        # 整理顺序
        for i in range(len(data)):
            op_ls = ans_op_ls[i]
            op_ls_len = len(op_ls)
            for j in range(op_ls_len):
                min_index = j
                for k in range(j,op_ls_len):
                    if(op_ls[min_index].timestamp > op_ls[k].timestamp):
                        min_index = k
                op_ls[j],op_ls[min_index] = op_ls[min_index],op_ls[j]
        # 添加子节点情况
        for line in data:
            id = line[0]
            op_ls = ans_op_ls[id-1]
            if(len(op_ls)==1): # 无路由节点
                for kid in op_list[id-1].children:
                    op_ls[0].children.append(ans_op_ls[kid[0]-1][0])
            else:
                for j in range(1,len(op_ls)): # 设置每一个路由PE的父子关系
                    father = op_ls[j-1]
                    kid = op_ls[j]
                    father.children.append(kid)
                for kid in op_list[id-1].children:
                    out_kid = ans_op_ls[kid[0]-1][0]
                    # 找到离此节点最近的父路由pe
                    fa_index = 0
                    for fa_op in op_ls:
                        if(fa_op.timestamp >= (out_kid.timestamp)):
                            break
                        else:
                            fa_index+=1
                    if(fa_index == len(op_ls)):
                        op_ls[fa_index-1].children.append(out_kid)
                    else:
                        op_ls[fa_index-1].children.append(out_kid)
        
        # 结果图表
        ans_table = np.zeros((ans_var_num,8))
        for out_op_ls in ans_op_ls:
            time_count = 0
            for out_op in out_op_ls:
                ans_table[out_op.id-1][6] = out_op.specie

                ans_table[out_op.id-1][0] = out_op.id
                ans_table[out_op.id-1][1] = out_op.timestamp
                for j in range(len(out_op.children)):
                    ans_table[out_op.id-1][2+j] = out_op.children[j].id
                ans_table[out_op.id-1][7] = out_op.old_id

        # 判断长依赖
        for line in ans_table:
            fa_time = int(line[1])
            for j in range(2,6):
                kid_id = int(line[j])
                if(kid_id == 0):
                    break
                else:
                    kid_time = int(ans_table[kid_id-1][1])
                    if ((kid_time-fa_time)%ii == 0):
                        optimal_flag = False
                        break
            
        
        if (optimal_flag == True):
            # 查看解的状态
            print("Status:", LpStatus[prob.status])
            # 查看解
            print("-------------------print answers-------------------")
            for v in prob.variables():
                if(v.name == "Npe"):
                    print(v.name, "=", v.varValue)
                if(v.varValue == 1):
                    print(v.name, "=", v.varValue)

            # 输出到文件
            output_file(ans_table)

            break


if __name__ == '__main__':
    scheduled(get_data(), 16)
