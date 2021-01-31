#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class Solution:

    # checkRecord
    def solve(self, n: int) -> int:
        MOD = 10 ** 9 + 7

        dp_0_0 = [0] * (n + 1)  # 长度为n 的序列中, 有0个A 并且结尾 是0个L
        dp_0_1 = [0] * (n + 1)  # 长度为n 的序列中, 有0个A 并且结尾 是1个L
        dp_0_2 = [0] * (n + 1)  # 长度为n 的序列中, 有0个A 并且结尾 是2个L

        dp_1_0 = [0] * (n + 1)  # 长度为n 的序列中, 有1个A 并且结尾 是0个L
        dp_1_1 = [0] * (n + 1)  # 长度为n 的序列中, 有1个A 并且结尾 是1个L
        dp_1_2 = [0] * (n + 1)  # 长度为n 的序列中, 有1个A 并且结尾 是2个L

        # 初始化
        dp_0_0[1] = 1
        dp_0_1[1] = 1
        dp_0_2[1] = 0

        dp_1_0[1] = 1
        dp_1_1[1] = 0
        dp_1_2[1] = 0

        for i in range(2, n + 1):
            dp_0_0[i] = (dp_0_0[i - 1] + dp_0_1[i - 1] + dp_0_2[i - 1]) % MOD
            dp_0_1[i] = dp_0_0[i - 1]
            dp_0_2[i] = dp_0_1[i - 1]

            dp_1_0[i] = (dp_0_0[i - 1] + dp_0_1[i - 1] + dp_0_2[i - 1] + dp_1_0[i - 1] + dp_1_1[i - 1] + dp_1_2[
                i - 1]) % MOD
            dp_1_1[i] = dp_1_0[i - 1]
            dp_1_2[i] = dp_1_1[i - 1]

        return (dp_0_0[n] + dp_0_1[n] + dp_0_2[n] + dp_1_0[n] + dp_1_1[n] + dp_1_2[n]) % MOD


class Test:
    def test_small_dataset(self, func):

        assert func(1) == 3

        assert func(2) == 8

        assert func(3) == 19



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



    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










