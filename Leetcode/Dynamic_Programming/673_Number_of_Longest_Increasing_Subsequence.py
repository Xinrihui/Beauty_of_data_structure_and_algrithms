#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:


    # findNumberOfLIS
    def solve(self, nums):
        """
        中间记录 LIS 的个数
        
        时间复杂度 

        :param s: 
        :return: 
        """
        L=len(nums)

        nums=[0]+nums # 数组 前面加上0 ,生成新的数组, 不改变原来的 nums

        dp=[1 for __ in range(L+1)] # LIS >=1
        count=[1 for __ in range(L+1)]

        dp[0]=0
        count[0] = 0

        for i in range(1,L+1):

            # max_length=float('-inf')

            for k in range(i-1,0,-1):

                if nums[i]>nums[k]:

                    current= dp[k]+1

                    if current > dp[i]: # 第1次找到 更长的 LIS
                        dp[i]=current
                        count[i]=count[k]

                    elif current == dp[i]: # 重复找到 一样长的 LIS
                        count[i]+=count[k]


        # print(dp)
        # print(count)

        Number=0 # LIS 的个数

        length=max(dp) # LIS 的长度

        for i in range(1, L + 1):
            if dp[i]==length:
                Number+=count[i]

        return Number

    # findNumberOfLIS
    def solve_naive(self, nums):
        """


        时间复杂度 

        :param s: 
        :return: 
        """
        L=len(nums)

        nums=[0]+nums # 数组 前面加上0 ,生成新的数组, 不改变原来的 nums

        # dp=np.zeros((L_s+1,L_s,L_s),dtype=bool) # 方便调试

        dp=[1 for __ in range(L+1)]

        dp[0]=0

        for i in range(1,L+1):
            for k in range(i-1,0,-1):
                if nums[i]>nums[k]:
                    dp[i]=max(dp[i],dp[k]+1)

        print(dp)

        # 追踪解
        self.Number=0 # LIS 的个数

        length=max(dp) # LIS 的长度
        prev_num=float('inf')

        dp_bound=L

        self.__process(nums,dp,dp_bound,length,prev_num) # TODO: TLE

        return self.Number

    def __process(self,nums,dp,dp_bound,length,prev_num):

        if length==0:

            self.Number+=1

            return

        for j in range(dp_bound,0,-1): #TODO： 递归里面 写for循环 你不超时 谁超时

            if dp[j]== length and nums[j]<prev_num:

                self.__process(nums,dp,j-1,length-1,nums[j])


class Test:
    def test_small_dataset(self, func):

        assert func([1,3,5,4,7]) == 2

        assert func([1,3,5,4,7,7]) == 4

        assert func([2,2,2,2,2]) == 5

        assert func([1, 2, 3, 4]) == 1

        assert func([4, 3, 2, 1]) == 4

        nums=[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30,
         29, 32, 31, 34, 33, 36, 35, 38, 37, 40, 39, 42, 41, 44, 43, 46, 45, 48, 47, 50, 49, 52, 51, 54, 53, 56, 55, 58,
         57, 60, 59, 61]
        assert func(nums) == 1073741824
        #
        # assert func("abcd") == 'a'

        # TODO: 边界条件
        assert func([1]) == 1

        # assert func('') == ''

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

        N = int(2 * pow(10, 3))
        max_v = int(pow(10, 6))

        l = np.random.randint(max_v, size=N)
        l=list(l)

        start = timeit.default_timer()
        print('run large dataset: ')
        func(l)
        end = timeit.default_timer()
        print('time: ', end - start, 's')


        # dir = 'large_test_case/188_1'
        # K, l1 = self.read_test_case_fromFile(dir)
        #
        # start = timeit.default_timer()
        # print('run large dataset:{} '.format(dir))
        # func(K, l1)  # 12.047259273 s
        # end = timeit.default_timer()
        # print('time: ', end - start, 's')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    # print(sol.solve([1,3,5,4,7]))

    print(sol.solve([1, 3, 5, 4, 7,10,9,12]))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    test.test_large_dataset(sol.solve)










