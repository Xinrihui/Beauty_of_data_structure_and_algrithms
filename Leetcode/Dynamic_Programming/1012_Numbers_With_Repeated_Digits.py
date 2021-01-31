#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class Solution(object):

    # numDupDigitsAtMostN
    def solve(self, n):
        """
        :type n: int
        :rtype: int
        """
        n_list = [int(ele) for ele in list(str(n))]  # [3,5,3]
        L = len(n_list)  # 3

        if L == 1:
            return 0

        # stage1
        Num1 = 0
        for i in range(1, L - 1 + 1):  # 1,2,..,L-1

            if i == 1:
                Num1 += 10

            else:
                times = 1
                for j in range(i):

                    if j == 0:
                        times = times * 9

                    else:
                        times = times * (10 - j)

                Num1 += times

        print('stage1 Num:', Num1)

        # stage2
        Num2 = 0
        i = 0
        while i < L - 1:  # 0,1,...,L-2

            for j in range(1, n_list[i]):  # 1,2,..,n_list[i]-1

                bits_num = L - i - 1  # 可以填空的位的数目

                print('i:{},j:{},bits_num:{}'.format(i, j, bits_num))

                prefix_list = n_list[0:i] + [j]
                print('prefix_list: ', prefix_list)

                if len(set(prefix_list)) != len(prefix_list):  # 前缀中存在 重复
                    continue

                if bits_num > 0:
                    times = 1
                    init = 10 - (i + 1)  #

                    for k in range(bits_num):  # 0,1,..,bits_num-1
                        times = times * (init - k)

                    Num2 += times

                    print('times:', times)

            # j 在边界情况的处理
            j = n_list[i]

            ii = i + 1
            # j=n_list[ii]

            if n_list[ii] != 0:

                bits_num = L - ii - 1  # 可以填空的位的数目

                # print('ii:{},j:{},bits_num:{}'.format(ii,j,bits_num))

                prefix_list = n_list[0:ii] + [0]
                # print('prefix_list: ',prefix_list)

                if len(set(prefix_list)) == len(prefix_list):  # 前缀中 不存在 重复

                    if bits_num > 0:
                        times = 1
                        init = 10 - (ii + 1)  #

                        for k in range(bits_num):  # 0,1,..,bits_num-1
                            times = times * (init - k)

                        Num2 += times

                        # print('times:',times)

            i += 1

        print('stage2 Num:', Num2)

        # stage3
        Num3 = 0

        start = n_list[0:L - 1] + [0]  # start=[3,5,0]
        end = n_list  # [3,5,3]

        for i in range(end[-1] + 1):  # 1,2,3

            prefix_list = start[0:L - 1] + [i]

            if len(set(prefix_list)) == len(prefix_list):
                Num3 += 1

        print('stage3 Num:', Num3)

        stage_sum = Num1 + Num2 + Num3

        return n + 1 - stage_sum


class Test:
    def test_small_dataset(self, func):

        assert func(353) == 84

        assert func(4353) == 1911

        assert func(105) == 11


        # TODO: 边界条件

        assert func(20) == 1


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

    # print(sol.solve(3))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










