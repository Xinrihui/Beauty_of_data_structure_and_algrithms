#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # maxProfit(self, prices: List[int]) -> int:
    def solve(self, prices ):
        """

        时间复杂度 O(n)

        :param s: 
        :return: 
        """
        n=len(prices)

        if n==0:
            return 0

        prices=[0]+prices

        # dp=(float('-inf'))*np.ones((n+1,3))

        dp=[[float('-inf'),float('-inf'),float('-inf')] for __ in range(n+1)]

        dp[1][0]=0
        dp[1][1] = -prices[1]

        for i in range(2,n+1):

            dp[i][0]=max(dp[i-1][0],dp[i-1][2])
            dp[i][1]= max(dp[i-1][1],dp[i-1][0]-prices[i])
            dp[i][2] = dp[i-1][1]+prices[i]

        res=max(dp[n][0],dp[n][2])

        return res

class Test:
    def test_small_dataset(self, func):

        prices =  [1,2,3,0,2]
        assert func(prices) == 3



        # TODO: 边界条件

        assert func([]) == 0

        assert func([1]) == 0



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
    prices = [1,2,3,0,2]
    print(sol.solve(prices))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










