#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

class Solution:

    # def largestSumOfAverages(self, A: List[int], K: int)
    def solve(self, A,K):
        """

        时间复杂度 

        :param s: 
        :return: 
        """
        n=len(A)

        A=[0]+A # 数组 前面加上0 ,生成新的数组, 不改变原来的 nums

        dp=np.zeros((n+1,K+1)) # 方便调试
        # dp= [[ 0 for __ in range(K+1)] for __ in range(n+1)] # 提高效率

        avg=np.zeros((n+1,n+1))
        # avg = [[0 for __ in range(n + 1)] for __ in range(n + 1)]

        # 预处理 计算所有 子数组的 平均数
        for i in range(1,n+1):
            for j in range(i, n + 1):

                if i==j:
                    avg[i][j]=A[i]
                else:
                    avg[i][j]= (avg[i][j-1]*(j-1-i+1)+A[j])/(j-i+1)

        print('avg:',avg)

        dp[1][1]=A[1]

        for i in range(2,n+1):
            for k in range(1,min(K,i)+1):

                if k==1:
                    # dp[i][k]= (dp[i-1][1]*(i-1)+A[i])/i
                    dp[i][k] =avg[1][i]

                elif k==i:
                    dp[i][k]= dp[i-1][i-1]+A[i]

                else:

                    for j in range(k-1,i):
                        dp[i][k]= max( dp[i][k] , dp[j][k-1] + avg[j+1][i] )

        print('dp:', dp)

        return dp[n][K]


class Test:
    def test_small_dataset(self, func):

        A = [9, 1, 2, 3, 9]
        K = 3
        assert func(A,K) == 20

        A = [1, 2, 3, 4, 5, 6, 7]
        K = 4
        assert func(A,K) == 20.5


        # TODO: 边界条件
        # assert func(None) == None
        #
        # assert func('') == ''


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

    A = [9, 1, 2, 3, 9]
    K = 3
    # print(sol.solve(A,K))

    A=[1, 2, 3, 4, 5, 6, 7]
    K=4
    # print(sol.solve(A, K))

    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










