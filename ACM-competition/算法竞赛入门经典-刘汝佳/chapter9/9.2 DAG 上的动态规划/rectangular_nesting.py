#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

import numpy as np

from collections import *

class Solution:
    """
    矩形嵌套问题
    
    ref: https://www.cnblogs.com/lepeCoder/p/7296052.html
    
    """

    def __granerateDAG(self,rec_list):
        """
        根据 矩形的边长, 生成 它们之间的嵌套关系的 DAG 图
        :param rec_list: 
        :return: 
        """

        N=len(rec_list) # 矩形的个数

        DAG={i:set() for i in range(N)} # 初始化 DAG

        for i in range(N):

            for j in range(i,N):

                # 矩形i 嵌套 在矩形j 中
                if (rec_list[i][0]< rec_list[j][0] and rec_list[i][1]< rec_list[j][1]) \
                        or ( rec_list[i][1]< rec_list[j][0] and rec_list[i][0]< rec_list[j][1]):

                    DAG[i].add(j) # i -> j

        return DAG


    def Kahn(self,graph):
        """
        宽度优先遍历 求 拓扑排序(DAG图中 的各个节点的依赖情况)
        
        :param graph: 
        :return: 
        """

        # 1.统计所有节点的入度
        in_degree=defaultdict(int)
        for source in graph: # 所有 节点的入度 初始化为 0
            in_degree[source]=0

        for source in graph:
            for target in graph[source]:
                in_degree[target]+=1

        # print(in_degree)

        # 2. 找出 入度为0 的节点放入 队列queue 中
        queue=deque()
        for node in in_degree:
            if in_degree[node]==0:
                queue.append(node)

        # 3. 输出 入度为0 的节点到结果集中，并将 相关联节点的入度 -1
        res=[]
        while len(queue)>0:

            current=queue.popleft()
            res.append(current)

            for node in graph[current]:
                in_degree[node]-=1

                if in_degree[node]==0:  # 入度为0 的节点加入 queue
                    queue.append(node)

        # 4. 判断是否存在环路
        flag=False
        if len(res)<len(graph): # 拓扑排序的顶点个数 小于 图的顶点个数，说明存在环路
            flag=True

        return flag,res

    def solve(self, rec):
        """
        1.将原问题 转换为 求 DAG 上的最长路径
        
        2.采用 递推的方法 解 DAG 上的最长路径

        :param rec: 
        :return: 
        """
        N=len(rec)

        # 1. 将矩形的相互嵌套关系 转换为 DAG 图
        DAG=self.__granerateDAG(rec)
        # print(DAG)

        # 2.拓扑排序, 得到 图中各个节点 做 松弛操作(最长路径) 的顺序
        _,nodes_order=self.Kahn(DAG)
        # print(nodes_order)

        # 3.按照 顺序对 节点进行 松弛操作(最长路径)
        distance=[0]*N

        pre_node=[None]*N # 记录 到达该节点的最长路径 的前驱节点

        for s_node in nodes_order: # s_node 边的起点

            for e_node in DAG[s_node]: # e_node 边的终点

                if distance[e_node] < distance[s_node]+1: # 从 s_node 到 e_node 的距离 更大了

                    distance[e_node] = distance[s_node]+1 # 松弛操作(最长路径)
                    pre_node[e_node]=s_node #记录 前驱节点

        # print(distance)
        # print(pre_node)

        # 4. 找到 最长路径 的终点节点
        max_ele = np.max(distance)#  全局的最大值

        max_idx_list = np.where(distance == max_ele) # 最大值的 索引列表

        first_idx=max_idx_list[0][0] # 第一个索引

        # 5. 找出整个最长路径
        longest_path=[]

        node=first_idx # 最长路径的终点

        while node !=None:

            longest_path.append(node)
            node=pre_node[node]


        return longest_path



class Test:
    def test_small_dataset(self, func):

        assert func() == ''

        # TODO: 边界条件
        assert func(None) == None

        assert func('') == ''

    def test_large_dataset(self, func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题

        Limits


        :param func: 
        :return: 
        """

        N = int(2 * pow(10, 4))
        max_v = int(pow(10, 9))

        l = np.random.randint(max_v, size=N)
        l1 = list(l)

        start = timeit.default_timer()
        print('run large dataset: ')
        func()
        end = timeit.default_timer()
        print('time: ', end - start, 's')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    rec=[
        (1,2),
        (2,4),
        (5,8),
        (6,10),
        (7,9),
        (3,1),
        (5,8),
        (12,10),
        (9, 7),
        (2, 2)
    ]

    print(sol.solve(rec))

    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










