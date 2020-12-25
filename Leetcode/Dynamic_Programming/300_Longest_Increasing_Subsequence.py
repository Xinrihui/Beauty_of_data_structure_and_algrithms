#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class Solution:

    #lengthOfLIS
    def solve(self, nums):

        n = len(nums)

        if n == 1:
            return 1

        nums = [-65536] + nums

        dp = [0 for __ in range(n + 1)]

        for i in range(1, n + 1):

            max_idx = 0
            max_length = 0

            for j in range(i - 1, -1, -1):

                # print('i:{},j:{}'.format(i,j))

                if nums[j] < nums[i] and dp[j] >= max_length:
                    max_idx = j
                    max_length = dp[j]

            dp[i] = dp[max_idx] + 1

        # print(dp)

        return max(dp)


class Test:
    def test_small_dataset(self, func):

        assert func([10,9,2,5,3,7,101,18]) == 4

        assert func([1,3,6,7,9,4,10,5,6]) == 6



        # TODO: 边界条件
        # assert func(None) == None

        assert func([1]) == 1


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

    # print(sol.solve([10,9,2,5,3,7,101,18]))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










