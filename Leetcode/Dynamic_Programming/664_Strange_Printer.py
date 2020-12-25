#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # strangePrinter(self, s: str) -> int:
    def solve(self, s):
        """
        记忆化递归

        时间复杂度 O(n^4)
        
        n<=100

        :param s: 
        :return: 
        """
        n=len(s)

        if n==0:
            return 0

        s = ' ' +s

        # self.dp=np.zeros((n+1,n+1,n+1),dtype=int)

        self.dp = [[[0 for __ in range(n+1)]for __ in range(n+1)]for __ in range(n+1)]

        i=1
        j=n
        k=0
        t=self.__process(s,i,j,k)

        return t

    def __process(self,s,i,j,k):
        """
        
        :param s: 
        :param i: 
        :param j: 
        :param k: 
        :return: 
        """

        if i>j:
            return 0
        if self.dp[i][j][k]>0:
            return self.dp[i][j][k]

        # 搜索剪枝
        while i< j-1 and s[j-1]==s[j]:
            j-=1
            k+=1

        case1= self.__process(s,i,j-1,0)+1

        case2=j-i+1  # 最多 能打印 j-i+1 次

        for p in range(i,j):

            if s[p]==s[j]:
                times= self.__process(s,i,p,k+1)+self.__process(s,p+1,j-1,0)
                case2=min(case2,times)

        res=min(case1,case2)

        self.dp[i][j][k]=res

        return res




class Test:
    def test_small_dataset(self, func):

        assert func("aaabbb") == 2

        assert func("aba") == 2

        assert func("abcccba") == 3


        # TODO: 边界条件
        assert func('') == 0

        assert func('a') == 1


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

    print(sol.solve("abcccba"))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










