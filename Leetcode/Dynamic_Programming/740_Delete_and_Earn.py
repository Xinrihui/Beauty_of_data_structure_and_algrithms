#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:


    def solve_naive(self, nums):
        """
        TLE 
        时间复杂度  O(L^2)=10^8

        :param nums: 
        :return: 
        """
        L= 10000+1

        freq=[0]*L
        dp=[0]*L

        for ele in nums:
            freq[ele]+=1

        dp[1]=1*freq[1]
        dp[2]=2*freq[2]

        for i in range(3,L):

            dp[i]=max(dp[0:i-1])+i*freq[i]  # TODO: O(L^2)

        res=max(dp)

        return res

    # deleteAndEarn(self, nums: List[int]) -> int:
    def solve(self, nums):
        """
        时间复杂度  O(L)
        
        :param nums: 
        :return: 
        """
        L= 10000+1

        freq=[0]*L
        dp=[0]*L

        for ele in nums:
            freq[ele]+=1

        dp[0]=0
        dp[1]=1*freq[1]
        dp[2]=2*freq[2]

        for i in range(3,L): #  O(L)

            dp[i]=max(dp[i-2],dp[i-3])+i*freq[i]

        res=max(dp)

        return res




class Test:
    def test_small_dataset(self, func):

        assert func([2, 2, 3, 3, 3, 4]) == 9

        assert func([3, 4, 2]) == 6



        assert func( [6, 5, 10, 2, 8, 6, 6, 5, 2, 9, 9, 4, 6, 3, 3, 7, 7, 8, 9, 5]) == 62

        # assert func("cbbd") == 'bb'
        #
        # assert func("cbbc") == 'cbbc'
        #
        # assert func("abcd") == 'a'

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


        l = [6,5,10,2,8,6,6,5,2,9,9,4,6,3,3,7,7,8,9,5]

        start = timeit.default_timer()
        print('run large dataset: ')
        func(l)
        end = timeit.default_timer()
        print('time: ', end - start, 's')


        # dir = 'large_test_case/188_1'
        # K, l1 = self.read_test_case_fromFile_list(dir)
        #
        # start = timeit.default_timer()
        # print('run large dataset:{} '.format(dir))
        # func(K, l1)  # 12.047259273 s
        # end = timeit.default_timer()
        # print('time: ', end - start, 's')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    # print(sol.solve([6,5,10,2,8,6,6,5,2,9,9,4,6,3,3,7,7,8,9,5]))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve_naive)










