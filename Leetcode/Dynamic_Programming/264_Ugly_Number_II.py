#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class Solution:



    # nthUglyNumber
    def solve(self, n: int):
        """
                    
        :param n: 
        :return: 
        """

        dp=[0]*(n+1)

        dp[1]=1 # 第1个 丑数是 1

        #  p2  p3 p5
        p2,p3,p5=1,1,1

        for i in range(2,n+1):

            print('i:',i)

            print('p2:{},p3:{},p5:{}'.format(p2, p3, p5))

            ugly2= 2*dp[p2]

            ugly3= 3*dp[p3]

            ugly5 = 5 * dp[p5]

            print('ugly2:{},ugly3:{},ugly5:{}'.format(ugly2,ugly3,ugly5))


            ugly_min =min(ugly2,ugly3,ugly5)

            dp[i]=ugly_min

            if ugly_min==ugly2:
                p2+=1

            if ugly_min==ugly3:
                p3 += 1

            if ugly_min == ugly5:
                p5 += 1

        print(dp)

        return dp[-1]




    # nthUglyNumber
    def solve_deprecated(self, n: int) :
        """
        TLE 
        
        :param n: 
        :return: 
        """

        dp = [False, True]

        # dp[1] = True

        time = 1
        i = 2

        while time < n:
            dp.append(None)

            two = False
            if i % 2 == 0:
                two = dp[i // 2]

            three = False
            if i % 3 == 0:
                three = dp[i // 3]

            five = False
            if i % 5 == 0:
                five = dp[i // 5]

            dp[i] = (two | three | five)

            if dp[i] == True:
                time += 1

            i += 1

        # print(dp)

        return i - 1


class Test:
    def test_small_dataset(self, func):

        assert func(10) == 12

        assert func(428) == 432000


        # TODO: 边界条件
        # assert func(None) == None

        assert func(1) == 1


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

        n=428
        start = timeit.default_timer()
        print('run large dataset: ')
        func(n)
        end = timeit.default_timer()
        print('time: ', end - start, 's')




if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    # print(sol.solve(10))


    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    test.test_large_dataset(sol.solve_deprecated) # time:  0.22019136500000003 s

    # test.test_large_dataset(sol.solve) # time:  0.016618458000000058 s










