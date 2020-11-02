#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

import numpy as np

from collections import *

class Solution_deprecated:


    # maximumRequests
    def solve_deprecated(self, n,requests):
        """
        思路1 (错误) 
        1.每一个 顶点的出度 和入度 并取最小值 ,
        2.对所有顶点的最小值 求和
        
        
        :param n: 
        :param requests: 
        :return: 
        """

        G_out=defaultdict(list)

        G_in=defaultdict(list)

        for edge in requests:
            G_out[edge[0]].append(edge[1])
            G_in[edge[1]].append(edge[0])

        # print(G_out)
        # print(G_in)


        # 计算每一个顶点的 出度 和 入度
        res_ok=0

        for node in G_out:

            # if len(G_out[node])==len(G_in[node]):
            #
            #     res_ok+= len

            res_ok+= min(len(G_out[node]),len(G_in[node]))

        return res_ok


    def solve2(self, n, requests):
        """
        1.找出 图中一条环路, 记录下 路径的长度 并将其 从图中删除
        直到 图中没有环路
        
        2. 所有环路 的路径长度 求和
        
        :param n: 
        :param requests: 
        :return: 
        """

        G_out={}

        # 由边 生成 带权的图
        for i in range(n): # 必要步骤, 否则 会缺失顶点
            G_out[i]={}

        for edge in requests:

            if edge[1] in G_out[edge[0]]:

                G_out[edge[0]][edge[1]]+=1 # 权值即为边的个数

            else:

                G_out[edge[0]][edge[1]]=1



        # G_out= {0: {1: 2}, 1: {0: 1, 2: 1}, 2: {0: 1}, 3: {4: 1}}
        #  0: {1: 2}  --- 0 节点 到1节点 有 1 条边
        #  1: {0: 1, 2: 1}  -- 1 节点 到0节点 有 1 条边
        #                   -- 1 节点 到 2节点 有 1 条边

        print(G_out)


        res_requst_num=0

        while True:

            # 1.找出 环路
            flag,circle_path=self.dfs(G_out) # (True, [0, 1, 0])

            print(flag,circle_path) # True [0, 2, 2]

            if flag==False:
                break

            #2. 截取 环路 中的 终点 与 起点相同的 作为 有效环路(circle_path_val)
            #   并 将有效环路的 长度加和到 res_requst_num 中

            i=0
            # while i < len(circle_path)-1:
            #     if circle_path[i]==circle_path[-1]:
            #         break
            #     i+=1

            circle_path_val=circle_path[i:]

            print('circle_path_val: {}'.format(circle_path_val))

            res_requst_num += (len(circle_path_val)-1)

            #3. 删除图中的环路
            for i in range(1,len(circle_path)):

                start_node= circle_path[i-1]
                end_node=circle_path[i]

                G_out[start_node][end_node]-=1 # 减去 1 条边

            print(G_out)

        return  res_requst_num

import itertools

class Solution_ref:

    def solve(self, n, requests):

        """

        ref: 
        https://leetcode-cn.com/problems/maximum-number-of-achievable-transfer-requests/solution/python3-zu-he-jian-zhi-si-lu-xiang-jie-cong-bao-li/


        :param n: 
        :param req: 
        :return: 
        """

        for k in range(len(requests), 0, -1):

            for c in itertools.combinations(range(len(requests)), k):  # 枚举所有组合的技巧:用标号 枚举

                degree = [0] * n  # 所有节点的 入度

                for i in c:
                    degree[requests[i][0]] -= 1
                    degree[requests[i][1]] += 1

                if not any(degree):  # 如果都为空、0、false，则返回false，如果不都为空、0、false，则返回true。
                    return k
        return 0

    def maximumRequests(self, n , requests):

        # 状态压缩,枚举每一种状态下,是否满足点的出入度相同,记录最大值即可
        ans = 0
        m = len(requests)
        bit = 1 << m
        for i in range(bit):
            temp = 0
            degree = [0] * n
            for j in range(m):
                if (i >> j) & 1:
                    temp += 1
                    degree[requests[j][0]] += 1
                    degree[requests[j][1]] -= 1
            tag = 1
            # 判断当前选择边是否满足
            for k in range(n):
                if degree[k] != 0:
                    tag = 0
                    break
            if tag:
                ans = max(ans, temp)
        return ans



class Solution:

    def __check_opt(self,n,edge_list):
        """
        判断 选取的 边的集合 是否满足 节点的 总的 出度 和 入度相等
        
        从 1500ms -> 840ms
        
        :param n: 
        :param edge_list: 
        :return: 
        """

        degree = [0] * n  # 所有节点的 入度

        for edge in edge_list:

            s=edge[0]
            t=edge[1]

            degree[s]-=1 # s 节点的 入度 -1
            degree[t]+=1 # t 节点的 入度 +1

        if not any(degree):  # 如果都为空、0、false，则返回false，如果不都为空、0、false，则返回true。
            return True

        return False



    def __check(self,edge_list):
        """
        判断 选取的 边的集合 是否满足 节点的 总的 出度 和 入度相等
        
        :param edge_list: 
        :return: 
        """

        node_degree=defaultdict(int)

        for edge in edge_list:

            s=edge[0]
            t=edge[1]

            node_degree[s]-=1 # s 节点的 入度 -1
            node_degree[t]+=1 # t 节点的 入度 +1

        for k,v in node_degree.items():

            if v!=0: # 节点的 入度必须为 0
                return False

        return True

    #   maximumRequests
    def solve_naive(self, n, requests):
        """
        暴力 求解
        
        :param n: 
        :param requests: 
        :return: 
        """

        # 枚举 所有的 边的集合 的情况
        for i in range(len(requests),0,-1): # 从最大的开始 倒着数, 相当于剪枝了

            for edge_list  in itertools.combinations(requests, i): # 边的集合(集合中的 元素个数为 i) 的所有可能情况

                # 判断 选取的 边 是否满足要求
                # if self.__check(edge_list) == True:

                if self.__check_opt(n,edge_list) == True:

                    return i

        return 0

    #   maximumRequests
    def solve(self,n, requests):
        """
        暴力 求解
        
        基于 状态压缩 (不使用 itertools.combinations )
        
        耗时：5280 ms
        
        :param n: 
        :param requests: 
        :return: 
        """

        m = len(requests)  # 6

        bit = 1 << m  # 2^6 = 64

        res=0

        for i in range(bit):  # 0,1,2,...,63

            # print("i:{}".format(i))

            degree = [0] * n

            one_num=0 # bin(i) 每一位为1的个数

            for j in range(m):

                if (i >> j) & 1:  # 判断每一位 是否为1

                    one_num+=1

                    degree[requests[j][0]] += 1
                    degree[requests[j][1]] -= 1

            if not any(degree):  # 如果都为空、0、false，则返回false，如果不都为空、0、false，则返回true。
                res = max(res, one_num)

        return res


class Test:
    def test_small_dataset(self, func):


        assert func(5, [[0, 1], [1, 0], [0, 1], [1, 2], [2, 0], [3, 4]]) == 5

        assert func(3,[[0, 0], [1, 2], [2, 1]]) == 3

        assert func(4,[[0,3],[3,1],[1,2],[2,0]]) == 4


        assert func(3, [[1, 2], [1, 2], [2, 2], [0, 2], [2, 1], [1, 1], [1, 2]]) == 4


        assert func(3, [[0, 0], [1, 1], [0, 0], [2, 0], [2, 2], [1, 1], [2, 1], [0, 1], [0, 1]]) == 5

        assert func(2, [[1,1],[1,0],[0,1],[0,0],[0,0],[0,1],[0,1],[1,0],[1,0],[1,1],[0,0],[1,0]]) == 11

        # TODO: 边界条件
        # assert func(None) == None
        #
        # assert func('') == ''

    def test_large_dataset(self, func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题

        Limits


        :param func: 
        :return: 
        """
        pass



if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    n = 5
    requests = [[0, 1], [1, 0], [0, 1], [1, 2], [2, 0], [3, 4]]

    print(sol.solve(n,requests))

    # print(sol.solve(n,requests))

    n=3
    requests=[[1, 2], [1, 2], [2, 2], [0, 2], [2, 1], [1, 1], [1, 2]]
    # print(sol.solve(n,requests))

    n=3
    requests=[[1, 2], [1, 2], [2, 2], [0, 2], [2, 1], [1, 1], [1, 2]]
    # print(sol.solve(n, requests))

    n=3
    requests =[[0, 0], [1, 1], [0, 0], [2, 0], [2, 2], [1, 1], [2, 1], [0, 1], [0, 1]]
    # print(sol.solve(n, requests))


    n =2
    requests =[[1, 1], [1, 0], [0, 1], [0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1], [0, 0], [1, 0]]
    # print(sol.solve(n, requests))

    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)













