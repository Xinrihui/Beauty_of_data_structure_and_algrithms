#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # ways(self, pizza: List[str], k: int) -> int:
    def solve(self, pizza,K):
        """

        时间复杂度 

        :param pizza: 
        :param K: 
        :return: 
        """
        m=len(pizza)
        n=len(pizza[0])

        # pizza_one=np.zeros((m+1,n+1),dtype=int)
        pizza_one=[[0 for __ in range(n+1)]for __ in range(m+1)] # 有苹果为1

        for i in range(1,m + 1):
            for j in range(1, n + 1):

                if pizza[i-1][j-1]=='A':
                    pizza_one[i][j]=1

        print(pizza_one)

        # 预处理
        # ones=np.zeros((m+1,n+1,m+1,n+1),dtype=int)
        ones=[[[[0 for __ in range(n+1)]for __ in range(m+1)] for __ in range(n+1)]for __ in range(m+1)]

        for i in range(1,m+1):
            for j in range(1,n+1):
                for s in range(1, m + 1):
                    for t in range(1, n + 1):

                        if i==s and j==t:
                            ones[i][j][s][t]=pizza_one[s][t]

                        elif i<s and j<t:
                            ones[i][j][s][t]= ones[i][j][s-1][t]+ones[i][j][s][t-1]\
                                              -ones[i][j][s-1][t-1]+pizza_one[s][t]

                        elif i<s and j==t:
                            ones[i][j][s][t] = ones[i][j][s-1][t]+pizza_one[s][t]

                        elif i==s and j<t:
                            ones[i][j][s][t] = ones[i][j][s][t-1] + pizza_one[s][t]

        # print(ones[1][1][2][2])
        # print(ones[1][1][2][3])
        # print(ones[2][2][3][3])

        # dp=np.zeros((m+1,n+1,K+1),dtype=int)
        dp = [[[0 for __ in range(K + 1)] for __ in range(n + 1)] for __ in range(m + 1)]

        for i in range(m,0,-1):
            for j in range(n,0,-1):
                for k in range(1, min(K,(m-i+1)*(n-j+1))+1):

                    # print('i:{} j:{} k:{}'.format(i,j,k))

                    if k==1:
                        if ones[i][j][m][n]>=1:
                            dp[i][j][k]=1
                        else:
                            dp[i][j][k] = 0

                    elif k>1:

                        v_slice=0
                        for v in range(j,n):
                            if ones[i][j][m][v]>=1:
                                v_slice+=dp[i][v+1][k-1]

                        h_slice=0
                        for h in range(i,m):
                            if ones[i][j][h][n]>=1:
                                h_slice+=dp[h+1][j][k-1]

                        dp[i][j][k] =v_slice+h_slice

        print(dp)

        return dp[1][1][K] % (pow(10,9)+7)


class Test:
    def test_small_dataset(self, func):

        pizza = ["A..", "AAA", "..."]
        k = 3
        assert func(pizza,k) == 3

        pizza = ["A..", "AA.", "..."]
        k = 3
        assert func(pizza, k) == 1

        pizza = ["A..", "A..", "..."]
        k = 1
        assert func(pizza, k) == 1


        # TODO: 边界条件

        pizza = ["A"]
        k = 1
        assert func(pizza, k) == 1


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
    pizza = ["A..", "AAA", "..."]
    k = 3
    print(sol.solve(pizza,k))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










