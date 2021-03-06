#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution_naive:

    #   def generateTrees(self, n: int) -> List[TreeNode]:
    def solve(self,  n):
        """
        回溯法

        时间复杂度 O(n!) 


        :param n: 
        :return: 
        """

        if n==0:
            return []

        left=1
        right=n

        res=self.__process(left,right)

        return res

    def __process(self, l,r):

        if l == r:
            return [TreeNode(l)]

        elif l>r:
            return [None]

        else: # l<r

            root_list=[]

            for i in range(l,r+1):
                left_list= self.__process(l,i-1)
                right_list= self.__process(i+1,r)

                for left in left_list:
                    for right in right_list:
                        root=TreeNode(i)
                        root.left=left
                        root.right=right

                        root_list.append(root)

        return root_list

class Solution:

    #   def generateTrees(self, n: int) -> List[TreeNode]:
    def solve(self,  n):
        """
        记忆化递归

        时间复杂度

        :param n: 
        :return: 
        """

        if n==0:
            return []

        left=1
        right=n

        self.dp={}

        res=self.__process(left,right)

        return res

    def __process(self, l,r):

        if (l,r) in self.dp:
            return  self.dp[(l,r)]

        if l == r:
            return [TreeNode(l)]

        elif l>r:
            return [None]

        else: # l<r

            root_list=[]

            for i in range(l,r+1):
                left_list= self.__process(l,i-1)
                right_list= self.__process(i+1,r)

                for left in left_list:
                    for right in right_list:
                        root=TreeNode(i)
                        root.left=left
                        root.right=right

                        root_list.append(root)

            self.dp[(l, r)]=root_list

            return root_list



class Test:
    def test_small_dataset(self, func):
        assert func(3, 4) == False

        assert func(10, 11) == False

        assert func(5, 7) == True

        assert func(6, 4) == True

        assert func(18, 79) == True

        assert func(5, 50) == False

        assert func(19, 190) == True

        # TODO: 边界条件
        # assert func(None) == None

        assert func(1, 0) == True

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

    sol = Solution_naive()

    # IDE 测试 阶段：

    print(sol.solve(3))


    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










