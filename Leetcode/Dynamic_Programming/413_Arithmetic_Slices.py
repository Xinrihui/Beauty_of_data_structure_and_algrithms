#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # numberOfArithmeticSlices
    def solve_opt(self, A) :
        """
        时间复杂度： O(n)
        
        :param A: 
        :return: 
        """

        n = len(A)
        dp = [0 for __ in range(n)]
        count=0

        for i in range(2,n):
            if A[i]-A[i-1]==A[i-1]-A[i-2]:
                dp[i]=dp[i-1]+1
                count+=dp[i]


        return count


    # numberOfArithmeticSlices
    def solve(self, A) -> int:
        """
        时间复杂度： O(n^2)
        
        :param A: 
        :return: 
        """

        n = len(A)

        dp = [[None for __ in range(n + 1)] for __ in range(n + 1)]

        count = 0

        for l in range(3, n + 1):
            for i in range(n - l + 1):

                j = i + l - 1

                # print('l:{},i:{},j:{}'.format(l,i,j))

                if l == 3:
                    if A[j] - A[i + 1] == A[i + 1] - A[i]:
                        dp[l][i] = A[i + 1] - A[i]
                        count += 1

                elif l > 3:
                    if A[j] - A[j-1] == dp[l - 1][i]:
                        dp[l][i] = dp[l - 1][i]
                        count += 1

        # print(dp)

        return count




class Test:
    def test_small_dataset(self, func):

        assert func([1,2,3,4]) == 3

        assert func([7,7,7,7]) == 3


        assert func([1,3,5,6,7,8]) == 4

        # assert func("cbbc") == 'cbbc'
        #
        # assert func("abcd") == 'a'

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

    print(sol.solve_opt([1,2,3,4]))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve_opt)

    # test.test_large_dataset(sol.solve)










