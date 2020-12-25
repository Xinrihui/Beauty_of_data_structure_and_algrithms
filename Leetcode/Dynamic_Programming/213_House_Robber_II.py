#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class Solution:

    def solve(self, nums):

        L = len(nums)

        if L==0:
            return 0

        if L == 1:
            return nums[0]

        return max(self.rob_basic(nums[1:]),self.rob_basic(nums[:-1]),self.rob_basic(nums[1:-1]))

    def rob_basic(self, nums):
        """
        基础的 打劫问题
        
        :param nums: 
        :return: 
        """

        L = len(nums)

        if L==0:
            return 0

        if L <= 2:
            return max(nums)

        dp = [0 for __ in range(L)]

        dp[0] = nums[0]
        dp[1] = nums[1]

        dp[2]= dp[0]+nums[2]

        for i in range(3, L):
            dp[i] = max(dp[i -2],dp[i-3]) + nums[i]

        # print(dp)

        return max(dp)


class Solution_deprecated:
    # rob
    def solve_wrong_answer(self, nums):
        """
        错误 

        :param nums: 
        :return: 
        """

        L = len(nums)

        if L == 1:
            return nums[0]

        dp = [0 for __ in range(L)]

        father = [0 for __ in range(L)]  # 记录 祖先节点, 首次打劫的店铺

        dp[0] = nums[0]
        dp[1] = nums[1]

        father[0] = 0
        father[1] = 1

        for i in range(2, L):
            # dp[i] = max(dp[0:i - 1]) + nums[i]

            max_v = float('-inf')
            max_idx = 0

            for j in range(i - 2, -1, -1):

                if dp[j] > max_v:
                    max_v = dp[j]
                    max_idx = j

            dp[i] = max_v + nums[i]
            father[i] = father[max_idx]

        print(dp)
        print(father)

        # 环形数组: 首尾 不能相连
        # 最后 1位的处理
        if father[-1] == 0:  # 首家店 被打劫

            max_v = float('-inf')

            i = L - 1
            for j in range(i - 2, -1, -1):

                if dp[j] > max_v and father[j] == 1:
                    max_v = dp[j]

            dp[i] = max(max_v + nums[i], dp[i] - min(nums[0], nums[i]))

            father[i] = 2

        print(dp)
        print(father)

        return max(dp)

class Test:
    def test_small_dataset(self, func):

        assert func([2, 3]) == 3

        # assert func([2,3,2]) == 3

        assert func([1, 3, 1, 3, 100]) == 103

        assert func([3, 2, 1, 3, 100]) == 102

        # assert func("cbbc") == 'cbbc'
        #
        # assert func("abcd") == 'a'

        # TODO: 边界条件
        assert func([2]) == 2


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

    print(sol.rob_basic([3, 2, 1, 3, 100]))

    # print(sol.solve([2,3,2]))

    # print(sol.solve([1, 3, 1, 3, 100]))


    # print(sol.solve([2, 1, 2, 6, 1, 8, 10, 10]))

    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










