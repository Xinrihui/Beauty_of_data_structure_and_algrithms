#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class Solution:


    # maxSubarraySumCircular
    def solve(self, A):
        """
         环形数组的 分情况讨论
         

        :param A: 
        :return: 
        """

        #1.最大子数组 不在 环路中
        not_circle_sub_sum=self.max_sub_array_opt(A)

        # 2. 最大子数组在环路中

        min_sub_sum=self.min_sub_array_opt(A)

        A_sum=sum(A)

        circle_sub_sum=A_sum-min_sub_sum

        if circle_sub_sum==0: # A 所有元素 都为 负数

            return not_circle_sub_sum

        return max(not_circle_sub_sum,circle_sub_sum)

    def min_sub_array_opt(self, nums):
        """
        最小 子数组

        优化 空间复杂度 为 O(1)

        :param nums: 
        :return: 
        """

        L = len(nums)

        if L==0:
            return 0
        elif L==1:
            return nums[0]

        dp_prev = nums[0]

        min_sum = float('inf')

        for i in range(1, L):
            dp = nums[i] + min(0, dp_prev)
            dp_prev = dp

            min_sum=min(min_sum,dp)


        return min_sum

    def max_sub_array_opt(self, nums):
        """
        最大 子数组
        
        优化 空间复杂度 为 O(1)
        
        :param nums: 
        :return: 
        """

        L = len(nums)

        if L==0:
            return 0
        elif L==1:
            return nums[0]

        dp_prev = nums[0]

        max_sum=float('-inf')

        for i in range(1, L):
            dp= nums[i] + max(0, dp_prev)
            dp_prev=dp

            max_sum=max(max_sum,dp)


        return max_sum

    def max_sub_array(self,nums):
        """
        最大 子数组
        
        :param nums: 
        :return: 
        """

        L=len(nums)

        dp = [0 for __ in range(L)]

        dp[0]=nums[0]

        for i in range(1, L):

            dp[i] = nums[i] + max(0, dp[i-1])

        print(dp)

        max_sum =  max(dp)

        return max_sum



class Solution_deprecated:

    # maxSubarraySumCircular
    def solve(self, A):
        """
        TLE 
        
        :param A: 
        :return: 
        """

        L = len(A)

        if L == 1:
            return A[0]

        A = A + A

        max_sum = float('-inf')

        sub_A = A

        for start in range(L): #

            dp = [0 for __ in range(L)]

            # sub_A = A[start:start + L]

            dp[0] = sub_A[start]

            for i in range(1, L):
                if dp[i - 1] > 0:
                    dp[i] = dp[i - 1] + sub_A[start+i]
                else:
                    dp[i] = sub_A[start+i]

            # print(dp)
            max_sum = max(max_sum, max(dp))

        return max_sum


class Test:
    def test_small_dataset(self, func):

        assert func([1,-2,3,-2]) == 3

        assert func([5,-3,5]) == 10

        assert func([3,-1,2,-1]) == 4

        assert func([3,-2,2,-3]) == 3

        assert func([-2,-3,-1]) == -1

        # TODO: 边界条件
        assert func([]) == 0

        assert func([2]) == 2

    def read_test_case_fromFile(self,dir):


        with open(dir,'r',encoding='utf-8') as file:  #

            # K1=int(file.readline().strip())
            # print('K1: ', K1)

            l=file.readline().strip()[1:-1].split(',')

            l= [int(ele) for ele in l]

            print('l:',l)

            return l

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

        # N = int(2 * pow(10, 4))
        # max_v = int(pow(10, 9))
        #
        # l = np.random.randint(max_v, size=N)
        # l1 = list(l)
        #
        # start = timeit.default_timer()
        # print('run large dataset: ')
        # func()
        # end = timeit.default_timer()
        # print('time: ', end - start, 's')


        dir = 'large_test_case/918'
        l = self.read_test_case_fromFile(dir)

        start = timeit.default_timer()
        print('run large dataset:{} '.format(dir))
        func(l)
        end = timeit.default_timer()
        print('time: ', end - start, 's')


if __name__ == '__main__':

    sol = Solution()

    sol_deprecated = Solution_deprecated()

    # IDE 测试 阶段：

    # print(sol.max_sub_array([5,-3,4]))

    # print(sol.max_sub_array_opt([5,-3,4]))

    # print(sol.min_sub_array_opt([5, -3, 4]))

    # print(sol.min_sub_array_opt([1,2,-3,2,-3,3]))

    # print(sol.solve([2]))

    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    test.test_large_dataset(sol_deprecated.solve) #  29.927291012999998 s
    test.test_large_dataset(sol.solve)  # time:  0.008831621000000012 s










