#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class Solution:

    # combinationSum4
    def solve(self, nums, target: int) -> int:
        """
        完全背包问题
        
        :param nums: 
        :param target: 
        :return: 
        """
        n = len(nums)

        nums = [0] + nums

        # 初始化
        dp = [0] * (target + 1)
        dp[0] = 1

        for i in range(1, target + 1):

            sum_cases = 0
            for k in range(1, n + 1):
                if i - nums[k] >= 0:
                    sum_cases += dp[i - nums[k]]

            dp[i] = sum_cases

        # print(dp)

        return dp[target]


class Solution2:
    """
    组合总和 Ⅳ+1
    
    nums 中每个元素 只能取1次 (因为 nums中有负数 若能取多次 则组合有 无穷多种, eg. (1,-1,1,-1,...))
    
    找出 和为给定目标正整数 的组合的个数 
    
    eg.
    nums = [1, 2, 3,-1]
    target = 4
    
    所有可能的组合为：
    (2,3,-1)
    (1,3)
    res=2
    
    eg.
    nums = [1, 2, -1,3]
    target = 2
    
    所有可能的组合为：
    (3,-1)
    (2)
    (1,2,-1)
    res=3
    
    limit:
    -100 <= nums[i] <=100
    target >= 0 
    
    """

    # combinationSum5
    def solve(self, nums, target: int) -> int:
        """
        01 背包问题
        
        :param nums: 
        :param target: 
        :return: 
        """
        n = len(nums)

        nums = [0] + nums

        # 初始化
        negSum=0
        for ele in nums:
            if ele<0:
                negSum+=ele

        N=target-negSum+1

        dp = [0] * (N + 1)
        dp[0] = 1

        for k in range(1, n + 1):

            if nums[k]>=0:

                for i in range(N,nums[k]-1,-1):
                    if i - nums[k] >= 0:
                        dp[i] += dp[i - nums[k]]


        for k in range(1, n + 1):

            if nums[k] < 0:

                for i in range(1,N+1):
                    # print('i:{},k:{},i - nums[k]:{}'.format(i,k,i - nums[k]))
                    if i - nums[k] <= N:
                        dp[i] += dp[i - nums[k]]


        print(dp)

        return dp[target]


class Test:
    def test_small_dataset(self, func):

        assert func([1, 2, 3],4) == 7


        # TODO: 边界条件
        # assert func(None) == None

        assert func([1],1) == 1

        assert func([1], 2) == 1


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


class Test2:
    def test_small_dataset(self, func):

        assert func([1, 2, 3,-1],4) == 2

        assert func([1, 2, -1, 3], 2) == 3

        assert func([1, 2, -1, 3], 0) == 1

        assert func([1, 2,0, -1, 3], 2) == 6

        assert func([1,3], 2) == 0

        assert func([10, -9], 1) == 1

        assert func([10, -8], 2) == 1

        assert func([-1,-2,6], 3) == 1

        assert func([-1,-2,6,4], 3) == 2

        # TODO: 边界条件
        # assert func(None) == None

        assert func([1],1) == 1

        assert func([1], 2) == 0

        # assert func([1], 2) == 1

if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：


    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)

    sol2=Solution2()
    # print(sol2.solve([1,2,3,-1],4))

    # print(sol2.solve([1, 2, -1, 3], 3))

    test2 = Test2()
    test2.test_small_dataset(sol2.solve)










