#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

class Solution:

    #   def mergeStones(self, stones: List[int], K: int) -> int:
    def solve(self, stones,K):
        """


        时间复杂度 O(n^3 * K)

        :param stones: 
        :return: 
        """
        n=len(stones)

        if (n-1) % (K-1)!=0:
            return -1

        stones=[0]+stones

        # 预处理
        prefix_sum=[0]*(n+1)
        for i in range(1,n+1):
            prefix_sum[i]=prefix_sum[i-1]+stones[i]


        # dp=np.zeros((n+1,n+1,K+1),dtype=int) # 方便调试
        dp=[[[0 for __ in range(K+1)] for __ in range(n+1)] for __ in range(n+1)] # 提高效率

        for l in range(1,n+1 ):
            for i in range(1, n-l+2 ):
                j=i+l-1

                for k in range(K,0,-1):
                    if i==j and k==1:
                        dp[i][j][k]=0
                    elif i==j and k>1:
                        dp[i][j][k] = float('inf')

                    elif k==1:
                        dp[i][j][k] = dp[i][j][K] + (prefix_sum[j]-prefix_sum[i-1])

                    elif k>1:

                        min_cost=float('inf')
                        for m in range(i,j): # 线性寻找最优切割点
                            cost= dp[i][m][1]+dp[m+1][j][k-1]
                            min_cost=min(min_cost,cost)

                        dp[i][j][k] =min_cost

        res=int(dp[1][n][1])

        return res


    def solve_opt1(self, stones,K):
        """
        
        优化时间复杂度
        
        由原本的线性找最优分割点 升级为 跳跃找 最优分割点
        
        时间复杂度 O( n^3 )

        :param stones: 
        :return: 
        """
        n=len(stones)

        if (n-1) % (K-1)!=0:
            return -1

        stones=[0]+stones

        # 预处理
        prefix_sum=[0]*(n+1)
        for i in range(1,n+1):
            prefix_sum[i]=prefix_sum[i-1]+stones[i]


        dp=[[[float('inf') for __ in range(K+1)] for __ in range(n+1)] for __ in range(n+1)] # 提高效率

        for i in range(1,n+1 ):
            dp[i][i][1] = 0

        for l in range(2,n+1):
            for i in range(1, n-l+2 ):
                j=i+l-1

                for k in range(2,K+1):

                    min_cost=float('inf')

                    for m in range(i,j,K-1): # 跳跃 寻找最优切割点
                        cost= dp[i][m][1]+dp[m+1][j][k-1]
                        min_cost=min(min_cost,cost)

                    dp[i][j][k] =min_cost

                dp[i][j][1] = dp[i][j][K] + (prefix_sum[j] - prefix_sum[i - 1]) #合并

        res= dp[1][n][1]

        return res


class Test:
    def test_small_dataset(self, func):

        stones = [3, 2, 4, 1]
        K = 2
        assert func(stones,K) == 20

        stones = [3, 2, 4, 1]
        K = 3
        assert func(stones, K) == -1

        stones = [3, 5, 1, 2, 6]
        K = 3
        assert func(stones, K) == 25



        # TODO: 边界条件
        # assert func(None) == None

        assert func([2],2) == 0


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

    print(sol.solve_opt1([3,5,1,2,6],3))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve_opt1)

    # test.test_large_dataset(sol.solve)










