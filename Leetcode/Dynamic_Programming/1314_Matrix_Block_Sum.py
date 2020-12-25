#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # matrixBlockSum(self, mat: List[List[int]], K: int) -> List[List[int]]:
    def solve(self, mat,K):
        """
        二维 前缀和 法

        时间复杂度 

        :param mat: 
        :param K: 
        :return: 
        """

        m=len(mat)
        n=len(mat[0])

        prefix= np.zeros((m,n),dtype=int) # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

        # prefix= [[0 for __ in range(n)]for __ in range(m)]

        for i in range(m):
            for j in range(n):
                a=0
                if i-1 >=0:
                    a=prefix[i-1][j]
                b=0
                if j-1>=0:
                    b=prefix[i][j-1]
                c=0
                if i-1>=0 and j-1>=0:
                    c= prefix[i-1][j-1]

                prefix[i][j]=a+b-c+mat[i][j]


        ans= np.zeros((m,n),dtype=int)
        # ans= [[0 for __ in range(n)]for __ in range(m)]

        for i in range(m):
            for j in range(n):

                a=max(0,i-K)
                b=max(0,j-K)

                c=min(m-1,i+K)
                d=min(n-1,j+K)

                sub1=0
                if b-1 >=0:
                    sub1=prefix[c][b-1]

                sub2=0
                if a-1>=0:
                    sub2=prefix[a-1][d]

                sub3=0
                if a-1>=0 and b-1>=0:
                    sub3= prefix[a-1][b-1]

                ans[i][j]= prefix[c][d]-(sub1+sub2-sub3)


        return ans

    # matrixBlockSum(self, mat: List[List[int]], K: int) -> List[List[int]]:
    def solve_v1(self, mat,K):
        """


        时间复杂度 

        :param mat: 
        :param K: 
        :return: 
        """

        m=len(mat)
        n=len(mat[0])

        mat_padding=np.zeros((m+2*K,n+2*K),dtype=int)

        for i in range(K,m+K):
            for j in range(K,n+K):

                mat_padding[i][j]=mat[i-K][j-K]

        # print(mat_padding)

        ans= [[0 for __ in range(n)]for __ in range(m)]

        pre_nums=None

        for i in range(K,n+K):

            if i==K: # 第1次
                nums=np.sum(mat_padding[:,i-K:i+K+1],axis=1)

            else:

                nums=pre_nums + mat_padding[:,i+K]- mat_padding[:,i-1-K]

            for j in range(K, m + K):

                if j==K:
                    ans[j-K][i-K]=sum(nums[0:j+K+1])

                else:
                    ans[j - K][i - K] = ans[j-K-1][i-K]+nums[j+K]-nums[j-1-K]


            pre_nums=nums


        return ans



class Test:
    def test_small_dataset(self, func):

        mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        K = 1
        assert func(mat,K) == [ [12,21,16],
                                 [27,45,33],
                                 [24,39,28]]
        mat = [[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]]
        K = 2
        assert func(mat, K) ==  [[45,45,45],[45,45,45],[45,45,45]]

        # TODO: 边界条件
        mat = [[1]]
        K = 1
        assert func(mat, K) == [[1]]

        mat = [[1]]
        K = 3
        assert func(mat, K) == [[1]]



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
    mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    K = 1

    print(sol.solve(mat,K))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










