#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # maxSumSubmatrix
    def solve(self, matrix, K):
        """
        滚动数组法
        
        时间复杂度：O(m*n^2)
        :param matrix: 
        :param K: 
        :return: 
        """

        matrix=np.array(matrix)

        m,n=matrix.shape # n=3

        max_sum=float('-inf')

        for start in range(n):

            rowSum=np.zeros(m,dtype=int)

            for end in range(start,n):

                # print('start:{},end:{}'.format(start,end))

                rowSum = rowSum + matrix[:,end]

                # print('rowSum:',rowSum)

                sub=self.max_upperbound_sub_array(rowSum,K) # rowSum 求 最大连续子数组, 并且 子数组的和 不超过 K

                # print('sub:', sub)

                # if sub<=K:
                max_sum=max(sub,max_sum)

        return max_sum


    def max_upperbound_sub_array(self, nums,K):
        """
        求最大 子数组的和，并且 子数组的和 不超过上界 K
        
        时间复杂度 O(n) 最坏情况： O(n^2)
        空间复杂度 O(n)

        :param nums: 
        :return: 
        """

        L = len(nums)

        if L==0:
            return float('-inf')

        elif L==1:

            if nums[0]<=K:

                return nums[0]

            else:
                return float('-inf')

        dp_prev = nums[0]

        max_sum=float('-inf')

        for i in range(1, L):
            dp= nums[i] + max(0, dp_prev)
            dp_prev=dp

            max_sum=max(max_sum,dp)

        if max_sum<=K:  # 找出的最大值比 上界小
            return max_sum

        # 暴力 遍历所有的 子数组
        max_sum = float('-inf')

        for start in range(L):

            sub_sum=0

            for end in range(start,L):

                sub_sum+=nums[end] # 滚动更新 子数组的值

                if sub_sum<=K:
                    max_sum=max(max_sum,sub_sum)

        return max_sum

    # maxSumSubmatrix
    def solve_naive(self, matrix,K):
        """
        暴力 动态规划 , 求出 所有矩形区域的值
        
        时间复杂度 O( m^2 * n^2 )
        
        TLE 

        :param matrix: 
        :return: 
        """
        m,n=len(matrix),len(matrix[0])

        # dp=np.zeros((m,n,m,n),dtype=int) # 方便调试

        dp=[[[[0 for __ in range(n)] for __ in range(m)] for __ in range(n)]for __ in range(m)] # 提高效率

        max_v=float('-inf')

        for k in range(m):
            for l in range(n):
                for i in range(k,m):
                    for j in range(l,n):

                        if i >0 and j>0:
                            dp[k][l][i][j]= dp[k][l][i][j-1]+dp[k][l][i-1][j]-dp[k][l][i-1][j-1] + matrix[i][j]

                        elif i==0 and j==0: # 边界 处理
                            dp[k][l][i][j] =matrix[i][j]

                        elif i==0:
                            dp[k][l][i][j] = dp[k][l][i][j - 1] + \
                                             matrix[i][j]
                        elif j==0:
                            dp[k][l][i][j] =  dp[k][l][i - 1][j]  + \
                                             matrix[i][j]

                        if dp[k][l][i][j]<=K:
                            max_v= max(max_v,dp[k][l][i][j])

        # print(dp)

        return max_v

class Test:
    def test_small_dataset(self, func):

        assert func([[2, 2, -1]],3) == 3

        matrix = [[1, 0, 1], [0, -2, 3]]
        k = 2
        assert func(matrix,k) == 2

        matrix = [[1, 0, 1],
                  [0, -2, 3],
                  [-2, 3, 1],
                  ]
        k=7
        assert func(matrix, k) == 6

        matrix=[[5, -4, -3, 4], [-3, -4, 4, 5], [5, 1, 5, -4]]
        k=10
        assert func(matrix, k) ==10

        matrix = [[5, -4, -3, 4],
                  [-3, -4, 4, 5],
                  [5, 1, 5, -4]]
        k = 8
        assert func(matrix, k) == 8

        # TODO: 边界条件
        # assert func(None) == None
        #
        # assert func('') == ''

    def read_test_case_fromFile_matrix(self,dir):
        """
        解析矩阵
        
        :param dir: 
        :return: 
        """


        with open(dir,'r',encoding='utf-8') as file:  #


            line_list=file.readline().strip()[2:-2].split('],[')

            matrix=[]

            for line in line_list:
                matrix.append([int(ele) for ele in line.split(',')])

            print('matrix:',matrix)

            K=int(file.readline().strip())
            print('K: ', K)

            return K,matrix

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


        dir = 'large_test_case/363'
        K, matrix = self.read_test_case_fromFile_matrix(dir)

        start = timeit.default_timer()
        print('run large dataset:{} '.format(dir))
        func(matrix, K)
        end = timeit.default_timer()
        print('time: ', end - start, 's')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    # matrix=[ [1,2,3,4],
    #          [-2,1,2,1],
    #          [3,2,1,-3],
    #          [2,1,1,1]
    #         ]

    matrix=[ [1,0,1],
             [0,-2,3],
             [-2,3,1],
            ]

    # print(sol.solve(matrix,7))

    matrix=[[2, 2, -1]]
    print(sol.solve(matrix, 3))

    matrix = [[5, -4, -3, 4],
              [-3, -4, 4, 5],
              [5, 1, 5, -4]]
    K = 10
    # print(sol.solve(matrix, K))

    matrix=[[5, -4, -3, 4],
             [-3, -4, 4, 5],
             [5, 1, 5, -4]]
    k=8
    # print(sol.solve(matrix, K))

    # print(sol.max_upperbound_sub_array([1,9,1],10))

    # print(sol.max_upperbound_sub_array([-2,-3,11], 8))

    print(sol.max_upperbound_sub_array([2, 2, -1], 3))

    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve) # time:  0.43965885699999996 s

    # test.test_large_dataset(sol.solve_naive) # time:  30.774332359000002 s










