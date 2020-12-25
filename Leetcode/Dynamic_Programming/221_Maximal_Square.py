#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class Solution:

    # maximalSquare
    def solve(self, matrix):

        if len(matrix)==0 or len(matrix[0])==0:
            return 0

        m, n = len(matrix), len(matrix[0])

        dp=np.zeros((m,n),dtype=int)

        # dp = [[0 for __ in range(n)] for __ in range(m)]

        i = 0
        for j in range(n):
            dp[i][j] = int(matrix[i][j])

        j = 0
        for i in range(m):
            dp[i][j] = int(matrix[i][j])

        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == '1':
                    dp[i][j] = max(1, min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])+1)

        # print(dp)

        max_length = np.max(dp)

        # max_length = 0
        # for i in range(m):
        #     max_length=max(max_length,max(dp[i]))


        max_area=max_length*max_length

        return max_area


class Test:
    def test_small_dataset(self, func):

        matrix = [["1", "0", "1", "0", "0"],
                  ["1", "0", "1", "1", "1"],
                  ["1", "1", "1", "1", "1"],
                  ["1", "0", "0", "1", "0"]]

        assert func(matrix) == 4

        matrix = [["1", "1", "1", "1", "0"],
                  ["1", "1", "1", "1", "0"],
                  ["1", "1", "1", "1", "1"],
                  ["1", "1", "1", "1", "1"],
                  ["0", "0", "1", "1", "1"]]

        assert func(matrix) == 16


        # TODO: 边界条件

        matrix = [["1"]]
        assert func(matrix) == 1

        assert func([]) == 0

        assert func([[]]) == 0

    def read_test_case_fromFile(self,dir):


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
        K, l1 = self.read_test_case_fromFile(dir)

        start = timeit.default_timer()
        print('run large dataset:{} '.format(dir))
        func(K, l1)  # 12.047259273 s
        end = timeit.default_timer()
        print('time: ', end - start, 's')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    matrix=[["1","0","1","0","0"],
            ["1","0","1","1","1"],
            ["1","1","1","1","1"],
            ["1","0","0","1","0"]]
    # print(sol.solve(matrix))

    matrix =[["1", "1", "1", "1", "0"],
             ["1", "1", "1", "1", "0"],
             ["1", "1", "1", "1", "1"],
             ["1", "1", "1", "1", "1"],
             ["0", "0", "1", "1", "1"]]

    # print(sol.solve(matrix))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










