#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    #  def numRollsToTarget(self, d: int, f: int, target: int) -> int:
    def solve(self, d: int, f: int, target: int):
        """

        时间复杂度 

        :return: 
        """

        # dp=np.zeros((d,target+1),dtype=int) # 方便调试

        dp=[[0 for __ in range(target+1)] for __ in range(d)] # 提高效率

        MOD=10**9+7

        # 初始化
        for t in range(1, min(f,target)+1):
            dp[0][t]=1

        for i in range(1,d):
            for t in range(1,target+1):

                sum=0
                for k in range(1,f+1):
                    if t-k >=0:
                        sum+=dp[i-1][t-k]
                        sum=sum % MOD

                dp[i][t]=sum

        # print(dp)

        return dp[-1][-1]


class Test:
    def test_small_dataset(self, func):

        d = 2
        f = 6
        target = 7
        assert func(d,f,target) == 6

        d = 1
        f = 6
        target = 3
        assert func(d,f,target) == 1

        d = 2
        f = 5
        target = 10
        assert func(d, f, target) == 1

        d = 1
        f = 2
        target = 3

        assert func(d, f, target) == 0

        d = 30
        f = 30
        target = 500
        assert func(d, f, target) == 222616187

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
        print('copy the test case to leetcode to judge the time complex')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：
    d = 2
    f = 6
    target = 7
    print(sol.solve(d,f,target))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










