#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # longestPalindrome
    def solve(self, s):
        """


        时间复杂度 

        :param s: 
        :return: 
        """
        # L_s=len(s)

        # s = ' ' + s
        # nums=[0]+nums # 数组 前面加上0 ,生成新的数组, 不改变原来的 nums

        # dp=np.zeros((L_s+1,L_s,L_s),dtype=bool) # 方便调试
        # dp=[[[False for __ in range(L_s)] for __ in range(L_s)] for __ in range(L_s+1)] # 提高效率

        # dp = [[[-65536,-65536] for __ in range(K + 1)] for __ in range(N + 1)] # TODO: 少掉 1层循环 时间减少很多 time


        # for l in range(1,L+1):
        #     for i in range(1,L-l+1+1):
        #         j=i+l-1
        #
        #         for k in range(i+1,j):
        #
        #             current = nums[i]*nums[k] * nums[j] + dp[k - i + 1][i]+ dp[j - k + 1][k]
        #
        #             dp[l][i]=max(dp[l][i],current) # 更新最大值


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
        print('copy the test case to leetcode to judge the time complex')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    print(sol.solve("babad"))


    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










