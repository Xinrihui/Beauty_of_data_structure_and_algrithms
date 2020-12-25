#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # minimumDeleteSum(self, s1: str, s2: str)
    def solve(self, s1, s2):
        """

        时间复杂度 

        :param s1: 
        :param s2: 
        :return: 
        """
        m=len(s1)
        n=len(s2)

        sum_s1=sum([ord(c) for c in s1])
        sum_s2=sum([ord(c) for c in s2])

        s1 = ' ' + s1
        s2 = ' ' + s2

        # dp=np.zeros((m+1,n+1),dtype=int) # 方便调试
        dp=[[0 for __ in range(n+1)] for __ in range(m+1)]  # 提高效率


        for i in range(1,m+1):
            for j in range(1,n+1):

                if s1[i]==s2[j]:
                    dp[i][j]=dp[i-1][j-1]+ord(s1[i])
                else:
                    dp[i][j] =max(dp[i-1][j],dp[i][j-1])

        print(dp)


        res=sum_s1 -dp[m][n]+ sum_s2-dp[m][n]

        return res



class Test:
    def test_small_dataset(self, func):

        s1 = "sea"
        s2 = "eat"
        assert func(s1,s2) == 231

        s1 = "delete"
        s2 = "leet"
        assert func(s1,s2) == 403



        # TODO: 边界条件


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

    print(sol.solve("delete","leet"))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










