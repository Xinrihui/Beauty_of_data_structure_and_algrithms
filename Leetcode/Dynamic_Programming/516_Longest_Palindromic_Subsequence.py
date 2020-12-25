#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class Solution:

    # longestPalindromeSubseq
    def solve_opt(self, s: str) -> int:
        """
        优化
        空间复杂度：O(n)

        :param s: 
        :return: 
        """

        n = len(s)

        s = ' ' + s

        dp1 = [0 for __ in range(n + 1)]
        dp2 = [0 for __ in range(n + 1)]
        dp3 = [0 for __ in range(n + 1)]

        for l in range(1, n + 1):

            dp3 = [0 for __ in range(n + 1)]

            for i in range(1, n - l + 2):

                j = i + l - 1

                if l == 1:
                    dp3[i] = 1

                elif l >= 2:
                    if s[i] == s[j]:
                        dp3[i] = dp1[i + 1] + 2
                    else:
                        dp3[i] = max(dp2[i], dp2[i + 1])
            dp1=dp2
            dp2=dp3

        res = dp3[1]

        return res

    #longestPalindromeSubseq
    def solve(self, s: str) -> int:
        """
        空间复杂度：O(n^2)

        :param s: 
        :return: 
        """

        n = len(s)

        s = ' ' + s

        # dp = [[0 for __ in range(n + 1)] for __ in range(n + 1)]

        dp=np.zeros((n+1,n+1),dtype=int)

        for l in range(1, n + 1):
            for i in range(1, n - l + 2):

                j = i + l - 1

                if l == 1:
                    dp[l][i] = 1

                elif l >= 2:
                    if s[i] == s[j]:
                        dp[l][i] = dp[l - 2][i + 1] + 2
                    else:
                        dp[l][i] = max(dp[l - 1][i], dp[l - 1][i + 1])

        print(dp)

        res = dp[n][1]

        return res


class Test:
    def test_small_dataset(self, func):

        assert func("bbbab") == 4

        assert func("cbbd") == 2

        assert func("abc") == 1


        # TODO: 边界条件
        assert func("a") == 1



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

    # print(sol.solve("bbbab"))

    print(sol.solve("abcdba"))

    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve_opt)

    # test.test_large_dataset(sol.solve)










