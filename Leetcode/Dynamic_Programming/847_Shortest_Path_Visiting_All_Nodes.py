#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    #  def shortestPathLength(self, graph: List[List[int]]) -> int:
    def solve(self, graph):
        """
        状态压缩的 BFS 

        时间复杂度 O(n*2^n)

        :param s: 
        :return: 
        """

        n=len(graph)

        visit=set()
        q=deque()
        end=(1<<n)-1

        for s in range(n):

            state= (1<<s)
            sNode=(s,state)

            visit.add(sNode)
            q.append(sNode)

        res=self.bfs(graph,visit,q,end)

        return res

    def bfs(self,G,visit,q,end):

        t=0

        while len(q)!=0:

            # t+=1
            L=len(q)

            for __ in range(L):

                node=q.popleft()
                c=node[0]
                state=node[1]

                if state==end:
                    return t

                for next in G[c]:

                    nextState= state | (1<<next)
                    nextNode=(next,nextState)

                    if nextNode not in visit:
                        visit.add(nextNode)
                        q.append(nextNode)

            t += 1

        return t

class Test:
    def test_small_dataset(self, func):

        assert func([[1],[0,2,4],[1,3,4],[2],[1,2]]) == 4

        assert func([[1,2,3],[0],[0],[0]]) == 4

        assert func([[1,2,3],[0,3],[0],[0,1]]) == 3


        # TODO: 边界条件
        # assert func(None) == None
        #
        assert func([[]]) == 0


    def read_test_case_fromFile_matrix(self, dir):
        """
        解析矩阵

        :param dir: 
        :return: 
        """

        with open(dir, 'r', encoding='utf-8') as file:  #


            line_list = file.readline().strip()[2:-2].split('],[')

            matrix = []

            for line in line_list:
                matrix.append([int(ele) for ele in line.split(',')])

            print('matrix:', matrix)

            K = int(file.readline().strip())
            print('K: ', K)

            return K, matrix

    def read_test_case_fromFile_list(self,dir):
        """
        解析 列表
        
        :param dir: 
        :return: 
        """


        with open(dir,'r',encoding='utf-8') as file:  #

            K1=int(file.readline().strip())
            print('K1: ', K1)

            l1=file.readline().strip()[1:-1].split(',')

            l1= [int(ele) for ele in l1]

            print('l1:',l1)

            return K1,l1

    def test_large_dataset(self, func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题

        Limits


        :param func: 
        :return: 
        """

        # RecursionError: maximum recursion depth exceeded in comparison
        # 默认的递归深度是很有限的（默认是1000）
        # import sys
        # sys.setrecursionlimit(100000)  # 设置 递归深度为 10w

        N = int(2 * pow(10, 4))
        max_v = int(pow(10, 9))

        l = np.random.randint(max_v, size=N)
        l1 = list(l)

        start = timeit.default_timer()
        print('run large dataset: ')
        func()
        end = timeit.default_timer()
        print('time: ', end - start, 's')


        dir = 'large_test_case/188_1'
        K, l1 = self.read_test_case_fromFile_list(dir)

        start = timeit.default_timer()
        print('run large dataset:{} '.format(dir))
        func(K, l1)  # 12.047259273 s
        end = timeit.default_timer()
        print('time: ', end - start, 's')
        print('copy the test case to leetcode to judge the time complex')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    print(sol.solve([[1,2,3],[0],[0],[0]]))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










