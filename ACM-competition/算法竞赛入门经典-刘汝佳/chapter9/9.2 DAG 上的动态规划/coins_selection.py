#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

import numpy as np

from collections import *


class Solution:
    """
    硬币凑面值问题

    ref: 

    """


    def Kahn(self, graph):
        """
        宽度优先遍历 求 拓扑排序(DAG图中 的各个节点的依赖情况)

        :param graph: 
        :return: 
        """

        # 1.统计所有节点的入度
        in_degree = defaultdict(int)
        for source in graph:  # 所有 节点的入度 初始化为 0
            in_degree[source] = 0

        for source in graph:
            for target in graph[source]:
                in_degree[target] += 1

        # print(in_degree)

        # 2. 找出 入度为0 的节点放入 队列queue 中
        queue = deque()
        for node in in_degree:
            if in_degree[node] == 0:
                queue.append(node)

        # 3. 输出 入度为0 的节点到结果集中，并将 相关联节点的入度 -1
        res = []
        while len(queue) > 0:

            current = queue.popleft()
            res.append(current)

            for node in graph[current]:
                in_degree[node] -= 1

                if in_degree[node] == 0:  # 入度为0 的节点加入 queue
                    queue.append(node)

        # 4. 判断是否存在环路
        flag = False
        if len(res) < len(graph):  # 拓扑排序的顶点个数 小于 图的顶点个数，说明存在环路
            flag = True

        return flag, res

    def __dp(self,current_value):
        """
        记忆化搜索 从节点 i(面值为 i) 出发到 节点0(面值为 0) 的最长路径
        
        :param current_value : 当前的面值为 
        :return: 
        """

        if current_value in self.visited:
            return self.distance[current_value]

        max_distance=float('-inf')

        for j in range(self.N):

            if current_value-self.coin_list[j] >= 0: # 若所有硬币都没法选择, 则 max_distance=float('-inf')

                d=self.__dp(current_value-self.coin_list[j])+1

                if d>max_distance:
                    max_distance=d

        self.distance[current_value]=max_distance
        self.visited.add(current_value)

        return max_distance

    def solve(self, coin_list, S):
        """
        
        1.将原问题 转换为 求 DAG 上的最长路径

        2.采用 递推方程 解 DAG 上的最长路径
        
        :param coin_list: 
        :param S: 
        :return: 
        """
        N = len(coin_list)

        # 1. 子问题 的划分
        distance = [0] * (S + 1) # distance[i] 从节点 i 出发 到 节点0 的最长距离

        #初始条件: distance[0]=0

        # 2. 子问题的求解顺序
        for i in range(1,S + 1): # i节点(面值为 i) i=1,2,3,..,9

            # 3. 递推方程
            max_distance = float('-inf')

            for j in range(N): # 选择不同的硬币

                if i - coin_list[j] >= 0: # 找不到合适的硬币, distance[i]=float('-inf')

                    if max_distance < distance[i-coin_list[j]]+1:

                        max_distance=distance[i-coin_list[j]]+1

            distance[i]=max_distance


        # 4.找出整个最长路径

        print(distance)

        longest_path_length = distance[S] # 从节点 S 出发到节点0 的最长路径的长度

        if longest_path_length== float('-inf'): # 从节点 S  无法到达 节点0
            return None

        start_node = S  # 起点

        current_distance = distance[start_node]

        longest_path = [start_node]

        while current_distance != 0: # 0 节点为终点

            for i in range(S + 1):

                if distance[i] == current_distance - 1:
                    longest_path.append(i)

                    current_distance = distance[i]

                    break

        return longest_path_length, longest_path


    def solve_recursion(self,coin_list,S):
        """
        1.将原问题 转换为 求 DAG 上的最长路径

        2.采用 记忆化搜索的方法(递归) 解 DAG 上的最长路径

        :param coin_list: 硬币的面值列表
        :param S: 要凑出的面值
        :return: 
        """
        self.N = len(coin_list)

        self.coin_list=coin_list

        # 1. distance[i] 表示 从节点 i 出发到 节点0 的最长路径的长度

        self.distance = [0] * (S+1)
        # self.pre_node = [None] * (S+1)  # 记录 节点的最长路径 的后继节点

        # 2. 记忆化搜索 找到从 节点S 出发 到节点0 的最长路径

        self.visited=set() # 记录已经访问的节点

        end_node=0 # 终点
        self.visited.add(end_node) # 递归到 0节点(面值为0), 应该结束递归

        start_node=S # 起点
        self.__dp(start_node)

        print(self.distance)

        longest_path_length= self.distance[start_node]

        # 5. 找出整个最长路径

        current_distance = self.distance[start_node]

        longest_path = [start_node]

        while current_distance!=0:

            for i in range(S+1):

                if self.distance[i] == current_distance-1:

                    longest_path.append(i)

                    current_distance=self.distance[i]

                    break

        return longest_path_length,longest_path






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

    coin_values=[2,3,5]

    # print(sol.solve_recursion(coin_values, 9))
    print(sol.solve(coin_values,9))

    # coin_values = [3, 5]
    # print(sol.solve(coin_values,7))


    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










