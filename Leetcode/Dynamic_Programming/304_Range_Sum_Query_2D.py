#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np



class NumMatrix:
    def __init__(self, matrix):
        """
        时间复杂度 O(m*n)
        空间复杂度 O(m*n)
        
        :param matrix: 
        """

        m = len(matrix)

        if m == 0: #
            return

        n = len(matrix[0])

        # self.dp=np.zeros((m,n),dtype=int)

        self.dp =[[0 for __ in range(n)] for __ in range(m)]


        for i in range(m):
            for j in range(n):

                if i == 0 and j == 0:
                    self.dp[i][j] = matrix[i][j]

                elif i >0  and j > 0:
                    self.dp[i][j] = self.dp[i - 1][j] + self.dp[i][j - 1] - \
                                          self.dp[i - 1][j - 1] + matrix[i][j]
                elif i > 0 and j == 0:
                    self.dp[i][j] = self.dp[i-1][j] + matrix[i][j]

                elif i == 0 and j > 0:
                    self.dp[i][j] = self.dp[i][j-1] + matrix[i][j]

        # print(self.dp)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:

        i,j,s,t=row1,col1,row2,col2

        left_region=0
        if j>0:
            left_region=self.dp[s][j-1]

        up_region=0
        if i>0:
            up_region = self.dp[i-1][t]

        left_up_region=0
        if i>0 and j>0:
            left_up_region= self.dp[i-1][j-1]


        return self.dp[s][t]-(left_region+up_region-left_up_region)

class NumMatrix_deprecated:
        def __init__(self, matrix):
            """
            时间复杂度 O(m^2*n^2)
            空间复杂度 O(m^2*n^2)

            :param matrix: 
            """

            m = len(matrix)

            if m == 0:
                return

            n = len(matrix[0])

            self.dp = [[[[0 for __ in range(n)] for __ in range(m)] for __ in range(n)] for __ in range(m)]

            for i in range(m):
                for j in range(n):
                    for s in range(m):
                        for t in range(n):

                            if i == s and j == t:
                                self.dp[i][j][s][t] = matrix[s][t]

                            elif i < s and j < t:
                                self.dp[i][j][s][t] = self.dp[i][j][s - 1][t] + self.dp[i][j][s][t - 1] - \
                                                      self.dp[i][j][s - 1][t - 1] + matrix[s][t]

                            elif i < s and j == t:
                                self.dp[i][j][s][t] = self.dp[i][j][s - 1][t] + matrix[s][t]

                            elif i == s and j < t:
                                self.dp[i][j][s][t] = self.dp[i][j][s][t - 1] + matrix[s][t]

        def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:

            return self.dp[row1][col1][row2][col2]


            # Your NumMatrix object will be instantiated and called as such:
            # obj = NumMatrix(matrix)
            # param_1 = obj.sumRegion(row1,col1,row2,col2)

# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)


class Test:
    def test_small_dataset(self, func):

        assert func(2,1,4,3) == 8

        assert func(1, 1, 2, 2) == 11

        assert func(1, 2, 2, 4) == 12


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

    matrix = [
        [3, 0, 1, 4, 2],
        [5, 6, 3, 2, 1],
        [1, 2, 0, 1, 5],
        [4, 1, 0, 1, 7],
        [1, 0, 3, 0, 5]
    ]

    sol=NumMatrix(matrix)

    # IDE 测试 阶段：

    print(sol.sumRegion(2,1,4,3))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.sumRegion)

    # test.test_large_dataset(sol.solve)










