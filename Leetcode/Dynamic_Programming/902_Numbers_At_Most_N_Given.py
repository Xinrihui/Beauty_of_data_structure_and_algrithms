#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

class Solution:

    # def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
    def solve(self, digits,n):
        """

    digits = ["1", "3", "5", "7"]
    n = 137

        时间复杂度 

        :param s: 
        :return: 
        """

        digits_int=[int(ele) for ele in digits]

        n_list = [int(ele) for ele in list(str(n)) ]

        # print('digits_int: ',digits_int)
        # print('n_list: ',n_list)

        digits_Num=len(digits_int)
        L=len(n_list)

        Nums=0

        # 位数: 1,2,..,L-1
        for i in range(1,L): # i - 位数
            Nums+= digits_Num**i

        #位数: L

        c=None
        t=0

        Nums+=self.__process(c,t,n_list,digits_int)

        return Nums

    def __process(self,c,t,n_list,digits_int):

        #递归结束条件
        if t< len(n_list):

            if c==None or c==n_list[t-1]:

                Nums = 0

                for ele in digits_int:

                    if ele<= n_list[t]:
                        Nums+= self.__process(ele,t+1,n_list,digits_int)

                    else: # 剪枝
                        break

                return Nums

            elif c < n_list[t-1]: # 剪枝: 当前位取的值 比 n_list对应 位小，则后几位 可以在 n_list 中任取

                return len(digits_int)**(len(n_list)-t)


        else: # t == len(n_list)

            return 1


class Test:
    def test_small_dataset(self, func):

        D = ["1", "4", "9"]
        N = 1000000000
        assert func(D,N) == 29523

        digits = ["1", "3", "5", "7"]
        n = 137
        assert func(digits,n) == 28


        digits = ["1", "3", "5", "7"]
        n = 100
        assert func(digits,n) == 20

        digits = ["1", "3", "5", "7"]
        n = 534
        assert func(digits,n) == 58

        digits = ["7"]
        n = 8
        assert func(digits, n) == 1

        digits = ["3", "4", "5", "6"]
        n = 64
        assert func(digits,n) == 18



        # TODO: 边界条件
        # assert func(None) == None



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
        print('copy the test case to leetcode to judge the time complex')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：


    # digits = ["1", "3", "5", "7"]
    # n = 137

    digits =["3", "4", "5", "6"]
    n = 64

    # print(sol.solve(digits,n))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










