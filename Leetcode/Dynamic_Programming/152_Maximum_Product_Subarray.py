#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

class Solution:

    def solve(self, nums):

        L = len(nums)

        nums = [0] + nums

        dp_pos = [0 for __ in range(L + 1)]
        dp_neg = [0 for __ in range(L + 1)]

        dp_pos[1] = nums[1]
        dp_neg[1] = nums[1]

        for i in range(2, L + 1):
            dp_pos[i] = max(nums[i], dp_pos[i - 1] * nums[i], dp_neg[i - 1] * nums[i])
            dp_neg[i] = min(nums[i], dp_pos[i - 1] * nums[i], dp_neg[i - 1] * nums[i])

        print(dp_pos)
        print(dp_neg)

        return max(dp_pos[1:])


class Test:
    def test_small_dataset(self, func):

        assert func([2,3,-2,4]) == 6

        assert func([2,3,-2,-1]) == 12

        assert func([-2,0,-1]) == 0

        # TODO: 边界条件
        # assert func(None) == None

        assert func([-2]) == -2

    def read_test_case_fromFile(self,dir):


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
        K, l1 = self.read_test_case_fromFile(dir)

        start = timeit.default_timer()
        print('run large dataset:{} '.format(dir))
        func(K, l1)  # 12.047259273 s
        end = timeit.default_timer()
        print('time: ', end - start, 's')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    print(sol.solve([2,3,-2,-1]))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










