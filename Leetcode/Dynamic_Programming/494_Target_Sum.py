#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

class Solution:

    # def findTargetSumWays(self, nums: List[int], S: int) -> int:
    def solve_opt(self, nums, S: int):
        """
        01 背包 

        状态压缩 -> 1维 动态规划
        
        降低空间复杂度

        时间复杂度 

        :return: 
        """

        sum_nums = sum(nums)  # 全部为 +

        if sum_nums < S:
            return 0

        # sum_nums >= S

        if (sum_nums - S) % 2 != 0:  # 无法整除
            return 0

        m = (sum_nums - S) // 2  # target

        n = len(nums)
        nums = [0] + nums

        dp = [0 for __ in range(m + 1)]

        dp[0] = 1

        for i in range(1, n + 1):
            for j in range(m ,nums[i]-1, -1):

                dp[j] += dp[j-nums[i]]

        # print(dp)

        res = dp[m]

        return res


    # def findTargetSumWays(self, nums: List[int], S: int) -> int:
    def solve(self, nums, S: int):
        """
        01 背包 
        
        2维 动态规划

        时间复杂度 

        :return: 
        """

        sum_nums=sum(nums) # 全部为 +

        if sum_nums < S:
            return 0

        # elif sum_nums==S: # nums 中可能有0
        #     return 1

        # sum_nums > S

        if (sum_nums-S) %2 !=0: # 无法整除
            return 0

        m= (sum_nums-S)//2 # target

        n=len(nums)
        nums=[0]+nums

        # dp=np.zeros((n+1,m+1),dtype=int)
        dp=[[0 for __ in range(m+1)]for __ in range(n+1)]

        dp[0][0]=1

        for i in range(1,n+1):
            for j in range(m+1):

                case1=0
                if j-nums[i]>=0:
                    case1=dp[i-1][j-nums[i]]
                case2=dp[i-1][j]
                dp[i][j]=case1+case2

        # print(dp)

        res=dp[n][m]

        return res







class Test:
    def test_small_dataset(self, func):

        assert func([1, 1, 1, 1, 1], 3) == 5

        assert func([1, 1],2) == 1

        assert func([1, 1],3) == 0


        assert func([1, 0], 1) ==2

        # TODO: 边界条件
        # assert func(None) == None

        assert func([1], 1) == 1

        assert func([1], 2) == 0


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

    # print(sol.solve([1, 1, 1, 1, 1], 3 ))

    # print(sol.solve([1, 0],1))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve_opt)

    # test.test_large_dataset(sol.solve)










