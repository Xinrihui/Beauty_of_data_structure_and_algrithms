#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

import numpy as np

from collections import *


class Solution:
    """
    无限 背包问题

    ref: 

    """



    def solve(self,weights,values,capacity,flag=False):
        """

        1.将原问题 转换为 求 带权 DAG 上的最长路径

        2.采用 递推方程 解 带权 DAG 上的最长路径

        :param weights: 
        :param values:
        :param capacity:
        :param flag: 是否要求背包的 容量一定要用完 
        
        :return: 
        """
        N = len(weights)

        # 1. 子问题 的划分
        distance = [0] * (capacity + 1)  # distance[i] 从节点 i 出发 到 节点0 的最长距离
                                         #             背包装入 i 重量的物品, 所能获得的最大价值

        pre_node = [None] * (capacity + 1) # 当背包重量为i , 并且获得最大价值时, 最后选择的物品标号

        # 初始条件: distance[0]=0

        # 2. 子问题的求解顺序
        for i in range(1, capacity + 1):  # i节点( 背包容量为 i) i=1,2,3,..,capacity

            # 3. 递推方程
            max_value = float('-inf')
            max_value_ele=-2

            for j in range(N):  # 选择不同的物品

                if i - weights[j] >= 0:  # 找不到合适的物品, distance[i]=float('-inf')

                    if max_value < distance[i - weights[j]] + values[j]:

                        max_value = distance[i - weights[j]] + values[j]
                        max_value_ele=j

            distance[i] = max_value
            pre_node[i] = max_value_ele

        # 4.整个最长路径

        print('distance:',distance)
        print('pre_node:',pre_node)

        if flag==True: # case1: 背包容量 必须用完, 即 最长路径的终点必须是 节点0

            longest_path_length = distance[capacity]  #  从节点 S 出发到节点0 的最长路径的长度

            if longest_path_length == float('-inf'):  # 从节点 S  无法到达 节点0
                return None
            current_weight = capacity  # 当前背包重量

        else: #case2: 背包的容量可以不用完, 即 最长路径的终点是 任意的
            longest_path_length = max(distance)#  全局的最大值
            current_weight = distance.index(longest_path_length) # 当前背包重量


        selected_ele_list = [] # 选择的物品列表

        while current_weight > 0:  # 0 节点为终点

            ele_index=pre_node[current_weight]

            selected_ele_list.append(ele_index)

            current_weight= current_weight - weights[ele_index]


        return longest_path_length, selected_ele_list




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

    weights = [2,3,4]

    values = [1,3,5]

    capacity=10

    print(sol.solve(weights,values,capacity))


    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










