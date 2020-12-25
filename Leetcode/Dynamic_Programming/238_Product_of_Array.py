#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class Solution:

    #productExceptSelf
    def solve(self, nums):
        """
        
        优化 空间复杂度 至 O(1)

        :param nums: 
        :return: 
        """

        n = len(nums)

        nums = [0] + nums

        output = [0] * (n + 2)

        output[0] = 1
        output[-1] = 1

        for i in range(n, 0, -1):
            output[i] = output[i + 1] * nums[i]

        pre_prefix=output[0]

        for i in range(1, n + 1):
            prefix= pre_prefix*nums[i]
            output[i] = pre_prefix * output[i + 1]

            pre_prefix=prefix

        return output[1: n+1]


    def solve_deprecated(self, nums ):
        """
        空间复杂度 O(n)
        
        :param nums: 
        :return: 
        """

        n = len(nums)

        nums = [0] + nums

        prefix = [0] * (n + 1)
        subfix = [0] * (n + 2)

        prefix[0] = 1
        subfix[-1] = 1

        output = [0] * (n + 1)

        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] * nums[i]

        for i in range(n, 0, -1):
            subfix[i] = subfix[i + 1] * nums[i]

        for i in range(1, n + 1):
            output[i] = prefix[i - 1] * subfix[i + 1]

        return output[1:]


class Test:
    def test_small_dataset(self, func):

        assert func([1,2,3,4]) == [24,12,8,6]



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


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










