#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # def lenLongestFibSubseq(self, A: List[int]) -> int:
    def solve(self, A):
        """

        时间复杂度 

        :param A: 
        :return: 
        """
        L=len(A)

        # dp=np.zeros((L,L),dtype=int) # 方便调试
        dp=[[0 for __ in range(L)] for __ in range(L)]  # 提高效率

        dict_A={}

        for i in range(L):
            dict_A[A[i]]=i

        # print(dict_A)

        # 1. 记录 斐波那契序列的 长度

        for i in range(1,L):
            for j in range(i+1,L):

                t=A[j]-A[i]

                # print('i:{},j:{},t:{}'.format(i,j,t))

                if t in dict_A and dict_A[t]<i:

                    if dp[dict_A[t]][i]==0:
                        dp[i][j]=3

                    else:
                        dp[i][j]=dp[dict_A[t]][i]+1

        # print(dp)

        # 2. 找到最长的 斐波那契数列的长度 并 记录 达到此长度的 数列的个数
        max_length=float('-inf')
        count=0

        for i in range(1,L):
            for j in range(i+1,L):

                if dp[i][j]==max_length:
                    count+=1

                if dp[i][j]>max_length:

                    max_length=dp[i][j]
                    count=1

        # print('count:',count)

        return max_length

class Test:
    def test_small_dataset(self, func):

        assert func([1,3,7,11,12,14,18]) == 3

        assert func([1,3,7,11,12,14,18,23]) == 4

        assert func([1,2,3,4,5,6,7,8]) == 5

        assert func([1,2,3]) == 3

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

    print(sol.solve([1,2,3,4,5,6,7,8]))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










