#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class ProductOfNumbers:
    def __init__(self):

        self.nums = [0]
        self.perfix = [1]

        self.N = 0

    def add(self, num: int) -> None:

        self.nums.append(num)
        self.N += 1

        if num != 0:
            pre_perfix = self.perfix[-1]
            self.perfix.append(pre_perfix * num)

        else:
            self.perfix = [0] * (self.N + 1)
            self.perfix[self.N] = 1

        print('nums: ',self.nums)
        print('N: ',self.N)
        print('perfix: ',self.perfix)

    def getProduct(self, k: int) -> int:

        if self.perfix[self.N - k] == 0:
            return 0

        else:
            return self.perfix[self.N] // self.perfix[self.N - k]


# Your ProductOfNumbers object will be instantiated and called as such:
# obj = ProductOfNumbers()
# obj.add(num)
# param_2 = obj.getProduct(k)


class Test:
    def test_small_dataset(self, func):

        assert func("babad") == 'bab'

        assert func("babab") == 'babab'

        assert func("cbbd") == 'bb'

        assert func("cbbc") == 'cbbc'

        assert func("abcd") == 'a'

        # TODO: 边界条件
        assert func(None) == None

        assert func('') == ''


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


    # IDE 测试 阶段：

    sol=ProductOfNumbers()
    sol.add(7)
    sol.add(5)
    sol.add(0)
    sol.add(4)
    sol.add(9)

    print(sol.getProduct(2))
    print(sol.getProduct(3))
    print(sol.getProduct(4))

    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










