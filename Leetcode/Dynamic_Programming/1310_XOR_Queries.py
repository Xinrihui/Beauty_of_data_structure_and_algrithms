#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
    def solve(self, arr,queries):
        """

        时间复杂度 

        :return: 
        """

        n=len(arr)

        prefix=[0]*n

        prefix[0]=arr[0]

        for i in range(1,n):
            prefix[i]=(prefix[i-1]^arr[i])

        res_list=[]

        for q in queries:
            i=q[0]
            j=q[1]

            if i==j:
                res_list.append(arr[i])
            else:
                if i-1>=0:
                    res_list.append( prefix[j]^prefix[i-1])
                else:
                    res_list.append(prefix[j] )


        return res_list

class Test:
    def test_small_dataset(self, func):

        arr = [1, 3, 4, 8]
        queries = [[0, 1], [1, 2], [0, 3], [3, 3]]

        assert func(arr,queries) == [2,7,14,8]

        arr = [4, 8, 2, 10]
        queries = [[2, 3], [1, 3], [0, 0], [0, 3]]
        assert func(arr, queries) == [8,0,4,4]

        # TODO: 边界条件
        # assert func(None) == None

        assert func([1],[[0,0]]) == [1]


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


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：
    arr = [1, 3, 4, 8]
    queries = [[0, 1], [1, 2], [0, 3], [3, 3]]

    print(sol.solve(arr,queries))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










