#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # def longestDecomposition(self, text: str) -> int:
    def solve(self, s):
        """

        时间复杂度: O(n^2)

        :param s: 
        :return: 
        """
        n=len(s)

        s = ' ' + s

        prev_i=1
        prev_j=n

        i=1
        j=n

        count=0
        while i<j:

            print('i:{},j:{}'.format(i,j))

            if s[prev_i:i+1]==s[j:prev_j+1]:

                print('left:{},right:{}'.format(s[prev_i:i+1],s[j:prev_j+1]))
                count+=2
                prev_i=i+1
                prev_j=j-1

            i+=1
            j-=1

        if prev_i <= prev_j:
            count+=1

        return count




class Test:
    def test_small_dataset(self, func):

        assert func("volalvo") == 5

        assert func("ghiabcdefhelloadamhelloabcdefghi") == 7

        assert func("antaprezatepzapreanta") == 11

        assert func("aaa") == 3

        assert func("abba") == 4


        # TODO: 边界条件
        assert func("a") ==1



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


    print(sol.solve("volalvo"))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










