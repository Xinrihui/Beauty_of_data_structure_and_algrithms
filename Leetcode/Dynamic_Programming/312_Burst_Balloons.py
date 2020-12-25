#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # maxCoins(self, nums: List[int]) -> int:
    def solve(self, nums):
        """
        区间 DP 

        时间复杂度 O(n^3)

        :param s: 
        :return: 
        """

        L=len(nums)

        if len(nums)==0:
            return 0

        nums=[1]+nums+[1] # 补上 前后边界

        L=L+2

        # print(nums)

        dp=[[0 for __ in range(L+1)] for __ in range(L+1)]

        # dp=np.zeros((L+1,L+1),dtype=int)

        for l in range(3,L+1):
            for i in range(0,L-l+1):
                j=i+l-1

                for k in range(i+1,j):

                    current = nums[i]*nums[k] * nums[j] + dp[k - i + 1][i]+ dp[j - k + 1][k]

                    dp[l][i]=max(dp[l][i],current) # 更新最大值


        # print(dp)

        res=dp[-1][0] # 前后 边界[1,...,1] 都不 戳破

        return res

    # maxCoins(self, nums: List[int]) -> int:
    def solve_naive(self, nums):
        """
        自顶向下递归 并缓存 子问题的解 

        时间复杂度 O(n!)

        :param s: 
        :return: 
        """
        self.dp={}

        if len(nums)==0:
            return 0

        res=self.__process(nums)

        return res

    def __process(self,nums):

        L = len(nums)

        if L == 1:
            return nums[0]

        if tuple(nums) in self.dp:
            return self.dp[tuple(nums)]
        
        
        if L==2:

            if nums[0]<= nums[1]:
                large=nums[1]
                small=nums[0]
            else:
                large = nums[0]
                small = nums[1]
            
            res=large*small+large

            self.dp[tuple(nums)]=res
            
            return res

        elif L>2:

            max_v=float('-inf')

            for idx in range(L):
            
                if idx==0:

                    v=nums[idx]*nums[idx+1]+self.__process(nums[1:])

                elif idx==L-1:

                    v = nums[idx] * nums[idx-1] + self.__process(nums[0:L-1])

                else:

                    v =nums[idx - 1]* nums[idx] * nums[idx+1] \
                    + self.__process(nums[:idx]+nums[idx+1:])

                max_v=max(max_v,v)

            self.dp[tuple(nums)] = max_v

            return max_v

class Test:
    def test_small_dataset(self, func):

        assert func([3, 1, 5, 8]) == 167

        assert func([4,1,2,3,10]) == 202

        assert func([4, 3, 5])==85

        assert func([1, 20, 3]) == 100

        assert func([4,1,2,3,1]) == 52


        # assert func("cbbc") == 'cbbc'
        #
        # assert func("abcd") == 'a'

        # TODO: 边界条件
        assert func([]) == 0

        # assert func([2]) == 2

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
        
        0 ≤ n ≤ 500, 0 ≤ nums[i] ≤ 100

        :param func: 
        :return: 
        """

        # RecursionError: maximum recursion depth exceeded in comparison
        # 默认的递归深度是很有限的（默认是1000）
        # import sys
        # sys.setrecursionlimit(100000)  # 设置 递归深度为 10w

        N = 500
        max_v = 100

        l = np.random.randint(max_v, size=N)
        l1 = list(l)

        start = timeit.default_timer()
        print('run large dataset: ')
        func(l1)
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

    # print(sol.solve_naive([3,1,5,8]))
    #
    print(sol.solve([3,1,5,8]))

    # print(sol.solve([1,20,3]))

    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    test.test_large_dataset(sol.solve)










