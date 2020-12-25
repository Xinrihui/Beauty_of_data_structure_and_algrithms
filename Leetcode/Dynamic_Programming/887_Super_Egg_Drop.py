#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    #  superEggDrop(self, K: int, N: int) -> int:
    def solve_opt(self, K, N):
        """
        利用 2分 查找 降低时间复杂度
        
        :param K: 
        :param N: 
        :return: 
        """
        self.dp = [[0 for __ in range(K+1)] for __ in range(N+1)]

        # self.dp= np.zeros((N+1,K+1),dtype=int) # 方便调试

        min_times=self.__process_opt(N,K) # 能确定 F 的最小移动次数

        # print(self.dp)

        return min_times

    def __process_opt(self, n, k):
        """
        
        有 k 个鸡蛋可以扔，在 共有 n 层楼的建筑中 扔鸡蛋，
        找到 最大的 让鸡蛋不会破的 楼层 F 所需要的 最少扔鸡蛋的次数 min_times
        
        利用 二分查找 优化 原来 1-n 的线性查找

        :param n: 
        :param k: 
        :return: min_times
        """

        if n == 1:
            return 1

        if n < 1:
            return 0

        if k == 1:
            return n

        if self.dp[n][k] == 0:  # __process(n,k) 还未被 计算

            min_times = float('inf')  # 很小的数

            # for i in range(1, n + 1):
            #     min_times = min(min_times, max(self.__process(i - 1, k - 1), self.__process(n - i, k)) + 1)

            left=1
            right=n

            while left <= right:

                mid= (left+right)//2

                broken= self.__process_opt(mid - 1, k - 1)
                not_broken= self.__process_opt(n - mid, k)

                if not_broken == broken:

                    min_times=broken+1

                    break

                elif not_broken < broken:

                    right=mid-1

                    min_times=min(min_times,broken+1) # 更新 min_times 为 大的那个

                elif not_broken > broken:

                    left=mid+1

                    min_times = min(min_times, not_broken+1)


            self.dp[n][k] = min_times

        else:  # __process(n,k) 已计算

            min_times = self.dp[n][k]

        return min_times

    #  superEggDrop(self, K: int, N: int) -> int:
    def solve(self, K, N):
        """
        
        自顶向下 递归
        
        时间复杂度 
        
        TLE 

        :param s: 
        :return: 
        """

        self.dp= np.zeros((N+1,K+1),dtype=int) # 方便调试

        min_times=self.__process(N,K) # 能确定 F 的最小移动次数

        print(self.dp)

        return min_times

    def __process(self,n,k):
        """
        有 k 个鸡蛋可以扔，在 共有 n 层楼的建筑中 扔鸡蛋，
        找到 最大的 让鸡蛋不会破的 楼层 F 所需要的 最少扔鸡蛋的次数 min_times
        
        :param n: 
        :param k: 
        :return: min_times
        """

        if n==1:
            return 1

        if n<1:
            return 0

        if k==1:
            return n

        if self.dp[n][k]==0: # __process(n,k) 还未被 计算

            min_times=float('inf') # 很小的数

            for i in range(1,n+1):

                min_times=min(min_times, max(self.__process(i-1,k-1) , self.__process(n-i,k))+1)

            self.dp[n][k]=min_times

        else:# __process(n,k) 已计算

            min_times=self.dp[n][k]

        return  min_times



class Test:
    def test_small_dataset(self, func):

        assert func(2,6) == 3

        assert func(1,2) == 2

        assert func(1, 3) == 3

        assert func(2, 3) == 2

        assert func(2, 5) == 3

        assert func(3,14) == 4

        assert func(1,10) == 10

        assert func(3,1) == 1

        # TODO: 边界条件
        # assert func(None) == None
        #
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

        K=4
        N=2000 # RecursionError: maximum recursion depth exceeded in comparison
               # 默认的递归深度是很有限的（默认是1000）

        import sys
        sys.setrecursionlimit(100000) # 设置 递归深度为 10w

        start = timeit.default_timer()
        print('run large dataset: ')
        func(K,N) #  12.513671179000001 s
        end = timeit.default_timer()
        print('time: ', end - start, 's')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    # print(sol.solve(2, 5))
    #
    # print(sol.solve_opt(2,5))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve_opt)

    test.test_large_dataset(sol.solve_opt)










