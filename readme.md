# 友好CGRA调度

##### 组员： 屈湘钧 李颀琳 许康乐

## 文件组成说明

mapping_friendly_CGRA_shcedule_memory.py	进行CGRA调度的python编程文件

example1.xls	研究报告输入例子

example2.xls	基于研究报告变更的自己例子

example3.xls	助教提供的测试样例图14

result1.csv	例子1的结果输出表

result2.csv	例子2的结果输出表

result3.csv	例子3的结果输出表

example1_new.PNG	例子一的手绘图(更改依赖约束后的图)

example1_old.PNG	例子一的有依赖约束问题手绘图

example2.PNG	例子二的手绘图

example3.PNG	例子三的手绘图

## 程序运行方式

在mapping_friendly_CGRA_shcedule_memory.py中直接运行main函数即可。

###### 函数调度路径：

​	mian	->	get_data()	->	scheduled()	->	output_file()

注意更换调度的输入数据文件在get_data中，输出文件变更在output_file中

## 输出表说明

节点类型列：0为算子节点 1 为插入的pe路由节点 2 为插入的store节点 3位插入的load节点