#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import *

import timeit

from collections import *

import copy

class Test:

    def test_small_dataset(self, func):


        assert func(5,[[1, 2], [2, 3], [3, 4], [2, 4], [5, 3]]) == '1 0 0 0 1'

        assert func(3,[[1,2],[3,2],[1,3]]) == '0 0 0'

        assert func(6, [[1, 2], [2, 3], [3, 4], [2, 4], [5, 3],[5,6]]) == '1 0 0 0 1 2'

        assert func(7, [[1, 2], [2, 3], [3, 4], [2, 4], [5, 3], [5, 6],[6,7]]) == '1 0 0 0 1 2 3'


    def test_large_dataset(self, func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题

        Limits


        :param func: 
        :return: 
        """

        N = int(2 * pow(10, 4))
        max_v = int(pow(10, 9))

        l = random.randint(max_v, size=N)
        l1 = list(l)

        start = timeit.default_timer()
        print('run large dataset: ')
        func()
        end = timeit.default_timer()
        print('time: ', end - start, 's')


class solutions:
    """

    采用 邻接矩阵 表示图

    时间复杂度过高, 导致 TTL 

    """

    def __transEdgeToGraph(self, N, edge_list):
        """
        将 输入的边 转换为 邻接矩阵 表示的图
        ( 节点标号 从 0 开始 )

        :param N: 
        :param edge_list: 
        :return: 
        """

        graph = zeros((N, N), dtype='uint8')

        for edge in edge_list:
            node1 = edge[0] - 1
            node2 = edge[1] - 1

            graph[node1][node2] = 1
            graph[node2][node1] = 1

        return graph

    def __topologicalSort(self, N, graph):
        """
        拓扑排序 输出 环所在的节点

         #[[0 1 0 0 0]
         # [1 0 1 1 0]
         # [0 1 0 1 1]
         # [0 1 1 0 0]
         # [0 0 1 0 0]]

        :param N: 
        :param graph: 
        :return: 
        """

        circle_nodes = set(list(range(N)))  # {0,1,2,3,4}

        while len(circle_nodes) != 0:

            for i in circle_nodes:

                edge_list = graph[i]

                #  所有点都至少有一条边, 图中没有孤立的点
                count = 0

                del_edge_right = -2

                for j in range(N):  # TODO: 每一个节点 都要去找 它邻接的节点, 时间复杂度 为 O(n^2) , 造成 TTL

                    if count > 1:  # 节点 的度 超过1, 无法删除
                        break

                    if edge_list[j] == 1:
                        del_edge_right = j
                        count += 1

                if count == 1:  # count==1 , 找到度为1 的节点，可以删除

                    del_edge_left = i
                    circle_nodes.remove(del_edge_left)

                    graph[del_edge_left][del_edge_right] = 0
                    graph[del_edge_right][del_edge_left] = 0

                    break  # 找到了 可以删除的节点, 执行删除后 跳出 for 循环


            else:  # 遍历图, 找不到可以 删除的节点

                break  # 说明有环路, 跳出 while 循环

        return circle_nodes

    def __cal_distance(self, N, node, circle_nodes, graph):
        """
        利用图的广度优先遍历，求 node 到环路的 最短距离

        :param N:
        :param node: 
        :param circle_nodes: 
        :param graph: 
        :return: 
        """

        queue = deque()
        queue.append(node)

        distance = 0

        while len(queue) != 0:

            current = queue.popleft()
            distance += 1

            for node in range(N):

                if graph[current][node] == 1:

                    if node in circle_nodes:

                        return distance

                    else:
                        queue.append(node)

        return None  # node 无法达到 图中的环路

    def solve(self, N, edge_list):

        graph = self.__transEdgeToGraph(N, edge_list)

        # [[0 1 0 0 0]
        # [1 0 1 1 0]
        # [0 1 0 1 1]
        # [0 1 1 0 0]
        # [0 0 1 0 0]]

        res_distance = zeros(N, dtype=int)

        nodes = set(list(range(N)))  # {0,1,2,3,4}

        graph_copy = graph.copy()

        circle_nodes = self.__topologicalSort(N, graph_copy)  # 环路中的 点的距离为 0

        nodes_not_in_circle = nodes - circle_nodes

        for node in nodes_not_in_circle:  # 不在环路中的点 要计算 与环路的最短距离

            res_distance[node] = self.__cal_distance(N, node, circle_nodes,
                                                     graph)  # TODO: 当环路中的节点很少时, 不在环路中的点的节点很多, 其中每一个点 都要 进行一次BFS, 造成 TTL

        res_distance = [str(ele) for ele in res_distance]

        return " ".join(res_distance)

class solutions_opt:
    """

    采用 邻接链表 表示图

    
    """

    def __transEdgeToGraph(self, edge_list):
        """
        将 输入的边 转换为 邻接链表 表示的图

        :param edge_list: 
        :return: 
        """

        graph = defaultdict(set)

        for edge in edge_list:

            node1 = edge[0]
            node2 = edge[1]

            graph[node1].add(node2)
            graph[node2].add(node1)

        return graph

    def __topologicalSort(self, graph_dict):
        """
        拓扑排序 输出 环所在的节点

        :param graph_dict: 
        :return: 
        """

        while len(graph_dict) != 0:

            for (start_node, edge_set) in graph_dict.items():

                # len(edge_set) 不可能为0 (所有点都至少有一条边, 图中没有孤立的点)

                if len(edge_set) == 1:  #  找到度为1 的节点，可以删除

                    end_node=edge_set.pop()

                    graph_dict[end_node].remove(start_node)

                    graph_dict.pop(start_node)

                    break  # 找到了 可以删除的节点, 执行删除后 跳出 for 循环


            else:  # 遍历图, 找不到可以 删除的节点

                break  # 说明有环路, 跳出 while 循环

        return graph_dict


    def __BFS_distance(self,circle_nodes,nodes_not_in_circle,graph_dict):
        """
        宽度优先搜索整个图 ,得到 图中每一个节点 与 图中环的距离
        
        :param circle_nodes: 
        :param nodes_not_in_circle: 
        :param graph_dict: 
        :return: 
        """

        res_distance = zeros(len(graph_dict) + 1, dtype=int)

        start= 0  # 0 号节点 未被使用

        # 1. 由于环路的各个点之间的距离可以看做0, 所以把环路中的所有节点 聚合为一个节点 start

        for s_node in circle_nodes:

            for e_node in graph_dict[s_node]:

                if e_node in nodes_not_in_circle: # 节点e_node 不是 环中的节点

                    graph_dict[start].add(e_node) # 节点 start 增加到 节点e_node的边

                    graph_dict[e_node].remove(s_node) # 节点e_node 删除到 s_node 的边

                    graph_dict[e_node].add(start) # 节点e_node 增加到 节点start 的边

            graph_dict.pop(s_node)

        # 2. 从 start 节点开始 宽度优先遍历图,
        #    由于环路中的所有节点距离都为0, 因此 以start 作为根节点进行 层次遍历 ,
        #    可以得到 每一个节点的深度( 每一个节点 与 start 的距离 即 与环路的距离)

        queue = deque()
        queue.append(start)
        distance = 0

        visited=set()

        while len(queue) != 0:

            L=len(queue)

            for i in range(L): # 层次遍历

                current = queue.popleft()
                res_distance[current]=distance

                visited.add(current) # 记录已经访问过的节点

                for node in graph_dict[current]:

                    if node not in visited: # 未访问过的节点 入队列
                        queue.append(node)

            distance+=1 # 下一层, 距离 +1

        return  res_distance


    def solve(self, N, edge_list):

        graph_dict = self.__transEdgeToGraph( edge_list)

        graph_copy = copy.deepcopy(graph_dict) # 深拷贝 graph_dict

        circle_nodes_dict = self.__topologicalSort( graph_copy)

        circle_nodes=circle_nodes_dict.keys() # 环路中的点与环路的最短距离为 0

        nodes = set(list(range(1, N + 1)))  # {1,2,3,4,5}

        nodes_not_in_circle = nodes - circle_nodes

        res_distance=self.__BFS_distance(circle_nodes,nodes_not_in_circle,graph_dict)

        # for node in nodes_not_in_circle: # 不在环路中的点 要计算 与环路的最短距离
        #     res_distance[node] = self.__cal_distance( node, circle_nodes, graph_dict)# TODO: 每一个点 都要 进行一次BFS, 造成 TTL


        res_distance = [str(res_distance[i]) for i in range(1,len(res_distance))] # res_distance 从标号1开始记录

        return " ".join(res_distance)



if __name__ == '__main__':

    sol = solutions_opt()

    flag=1# TODO:提交代码记得修改 状态

    # IDE 测试 阶段：
    if flag==1:

        # N=5
        # edge_list=[[1,2],[2,3],[3,4],[2,4],[5,3]]
        # print(sol.solve(N,edge_list))

        test = Test()
        test.test_small_dataset(sol.solve)

        # test.test_large_dataset(sol.solve)


    # 提交 阶段：
    elif flag == 2:

    # pycharm 开命令行 提交
    # E:\python package\python-project\Beauty_of_data_structure_and_algrithms\kickstart\2018 Round C>python planet_distance.py < inputs_PD
    # Case #1: 4

        T = int(input().strip())  # 一共T个 测试数据

        for t in range(1, T + 1):

            N = int(input().strip())

            edge_list=[]

            for j in range(N):
                edge = [ int(i) for i in input().split(' ') ]
                edge_list.append( edge )


            res = sol.solve(N,edge_list)

            print('Case #{}: {}'.format(t, res))









