#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

class Solution:

    # def knightDialer(self, n: int) -> int:
    def solve(self, N):
        """


        时间复杂度 

        :param n: 
        :return: 
        """

        if N==1:
            return 10

        L=5

        MOD=10**9 + 7

        # dp=np.zeros((N+1,L+1),dtype=int) # 方便调试
        dp=[[0 for __ in range(L+1)] for __ in range(N+1)]  # 提高效率

        # N=1 初始化
        dp[1][0]=2
        dp[1][1]=4
        dp[1][2]=1
        dp[1][3]=2
        dp[1][4]=9

        for n in range(2,N+1):

            dp[n][0] = dp[n-1][1]+2*dp[n-1][2]
            dp[n][1] = (dp[n-1][0]+dp[n-1][3])*2
            dp[n][2] = dp[n-1][0]
            dp[n][3] = dp[n-1][1]

            dp[n][4] = (3*dp[n-1][0]+2*(dp[n-1][4]-dp[n-1][0])) % MOD

        res=dp[N][4]

        return res


class Test:
    def test_small_dataset(self, func):

        assert func(2) == 20

        assert func(3) == 46

        assert func(4) == 104

        assert func(5) == 240


        # TODO: 边界条件
        assert func(1) == 10


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

    print(sol.solve(2))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










