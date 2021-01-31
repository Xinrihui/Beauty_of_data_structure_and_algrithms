#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

import math


class Solution(object):

    def combinationNum(self, n, m):
        """
        (4,2)= (4*3)/(2*1) 
        
        计算 n! : math.factorial(n)

        """

        return math.factorial(n) // (math.factorial(m) * math.factorial(n - m))

    def countOne(self, nums):
        """
        nums 中1的个数
        """

        count = 0
        for ele in nums:
            if ele == 1:
                count += 1

        return count

    # countDigitOne
    def solve(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return 0

        n_list = [int(ele) for ele in list(str(n))]  # [2,0,1,5]
        L = len(n_list)  # 4

        if L == 1:
            return 1

        # stage1
        Num1 = 0

        for i in range(1, L):  # 可填充的位数 1,2,3

            for t in range(1, i + 1):  # 填充的1 的 个数 : 1 个 1 ,  2 个 1 ,3 个 1

                # 首位为1
                not_one = i - t  # 填充 非1的位的 个数

                one = t - 1  # 除了首位，其他位填入1的位的 个数

                # combinationNum(i-1,one) 除了 首位的1, 剩下的1 要在 首位后面的 i-1位 中选
                num_first_one = self.combinationNum(i - 1, one) * (9 ** not_one)

                # 首位 不为1
                if not_one > 0:
                    # combinationNum(i-1,t) 首位不能填1, 1只能在 后面的 i-1 位中选择
                    num_first_not_one = self.combinationNum(i - 1, t) * 8 * (9 ** (not_one - 1))

                else:
                    num_first_not_one = 0

                Num1 += (num_first_one + num_first_not_one) * t

        print('stage1:', Num1)

        # 　stage2
        Num2 = 0

        for i in range(0, L - 1):  # 0,1,2

            for j in range(1, n_list[i]):  # 1,2,..., n_list[i]-1

                bits_num = L - i - 1  # 可填充的位数

                prefix = n_list[0:i] + [j]

                print('i:{},j:{},bits_num:{}'.format(i, j, bits_num))
                print('prefix:', prefix)

                if bits_num > 0:

                    prefix_one_num = self.countOne(prefix)  # 前缀中 1 的个数

                    for t in range(0, bits_num + 1):  # 填充的1 的 个数 : 0,1,..,bits_num

                        times = (self.combinationNum(bits_num, t) * (9 ** (bits_num - t))) * (t + prefix_one_num)
                        Num2 += times

                        print(times)

                        # j 在边界情况的处理
            j = n_list[i]
            ii = i + 1
            if n_list[ii] > 0:

                bits_num = L - ii - 1  # 可填充的位数

                prefix = n_list[0:ii] + [0]

                print('ii:{},j:{},bits_num:{}'.format(ii, j, bits_num))
                print('prefix:', prefix)

                if bits_num > 0:
                    prefix_one_num = self.countOne(prefix)  # 前缀中 1 的个数

                    for t in range(0, bits_num + 1):  # 填充的1 的 个数 :

                        times = (self.combinationNum(bits_num, t) * (9 ** (bits_num - t))) * (t + prefix_one_num)
                        Num2 += times

                        print(times)

        print('stage2:', Num2)

        # stage3
        Num3 = 0

        prefix = n_list[0:L - 1]
        prefix_one_num = self.countOne(prefix)  # 前缀中 1 的个数

        bits_num = 1

        if n_list[L - 1] == 0:
            Num3 += prefix_one_num

        else:
            for t in range(0, bits_num + 1):  # 填充的1 的 个数 :

                times = ((n_list[L - 1]) ** (bits_num - t)) * (t + prefix_one_num)

                Num3 += times

                # print(times)

        print('stage3:', Num3)

        return Num1 + Num2 + Num3


class Test:
    def test_small_dataset(self, func):

        assert func(2015) == 1608

        assert func(100) == 21

        assert func(1100) == 422

        # TODO: 边界条件

        assert func(0) == 0

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

    def read_test_case_fromFile_list(self, dir):
        """
        解析 列表

        :param dir: 
        :return: 
        """

        with open(dir, 'r', encoding='utf-8') as file:  #

            K1 = int(file.readline().strip())
            print('K1: ', K1)

            l1 = file.readline().strip()[1:-1].split(',')

            l1 = [int(ele) for ele in l1]

            print('l1:', l1)

            return K1, l1

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



    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










