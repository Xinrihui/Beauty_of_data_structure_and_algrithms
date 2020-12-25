#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # findLength(self, A: List[int], B: List[int]) -> int:
    def solve(self, A,B):
        """

        时间复杂度 

        :param A: 
        :param B:
         
        :return: 
        """
        m=len(A)
        n = len(B)

        A=[-1]+A # 数组 前面加上0 ,生成新的数组, 不改变原来的 nums
        B = [-1] + B

        # dp=np.zeros((m+1,n+1),dtype=int) # 方便调试
        dp=[[0 for __ in range(n+1)] for __ in range(m+1)]  # 提高效率


        for i in range(1,m+1):
            for j in range(1,n+1):

                if A[i]==B[j] and A[i-1]==B[j-1]:
                    dp[i][j]=dp[i-1][j-1]+1

                elif A[i]==B[j] and A[i-1]!=B[j-1]:
                    dp[i][j] =1

        max_len=float('-inf')

        for i in range(1, m + 1):
            max_len=max(max_len,max(dp[i]))


        return max_len


class Test:
    def test_small_dataset(self, func):

        assert func("babad") == 'bab'

        assert func("babab") == 'babab'

        assert func("cbbd") == 'bb'

        assert func("cbbc") == 'cbbc'

        assert func("abcd") == 'a'

        # TODO: 边界条件
        assert func(None) == None

        assert func('') == ''


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
    A= [1, 2, 3, 2, 1]
    B= [3, 2, 1, 4, 7]
    print(sol.solve(A,B))


    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










