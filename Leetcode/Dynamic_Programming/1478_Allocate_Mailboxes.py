#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # minDistance(self, houses: List[int], k: int) -> int:
    def solve(self, houses ,K):
        """

        时间复杂度 

        :param houses: 
        :return: 
        """
        houses=sorted(houses) # 找中位数 必须为排好序的数组

        # 预处理

        n=len(houses)

        houses=[0]+houses

        mid=np.zeros((n+1,n+1),dtype=int)

        # mid=[[0 for __ in range(n+1)]for __ in range(n+1)]

        for i in range(1,n+1):
            for j in range(i,n+1):

                median=houses[(i+j)//2 ]

                for k in range(i,j+1):
                    mid[i][j]= mid[i][j]+abs(houses[k]-median)

        print(mid)

        # 划分子问题
        dp = (float('inf'))*np.ones((n + 1, K + 1))

        # dp=[[float('inf') for __ in range(K+1)]for __ in range(n+1)]

        for i in range(1,n+1):
            for k in range(1,min(i,K)+1):

                if k==1:
                    dp[i][k]=mid[1][i]

                elif k==i:
                    dp[i][k]=0

                else:
                    for j in range(k-1,i):
                        dp[i][k]=min(dp[i][k],dp[j][k-1]+mid[j+1][i])
        print(dp)

        return int(dp[n][K])

class Test:
    def test_small_dataset(self, func):

        houses = [1, 4, 8, 10, 20]
        k = 3
        assert func(houses,k) == 5

        houses = [2,3,5,12,18]
        k = 2
        assert func(houses,k) == 9

        houses =   [7,4,6,1]
        k = 1
        assert func(houses,k) == 8

        houses =  [3,6,14,10]
        k = 4
        assert func(houses,k) == 0

        houses = [1, 8, 12, 10, 3]
        k = 3
        assert func(houses, k) == 4

        # TODO: 边界条件
        assert func([1],1) == 0



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
    houses = [1, 4, 8, 10, 20]
    k = 3
    # print(sol.solve(houses,k))


    houses=[1, 8, 12, 10, 3]
    k=3
    # print(sol.solve(houses, k))

    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










