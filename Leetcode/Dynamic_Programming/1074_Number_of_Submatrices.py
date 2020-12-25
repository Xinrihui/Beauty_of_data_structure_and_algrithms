#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *


import numpy as np
class Solution:

    # def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
    def solve(self, matrix, target):
        """

        时间复杂度 
        O(m*n^2)

        :param s: 
        :return: 
        """

        m=len(matrix)
        n=len(matrix[0])

        matrix=np.array(matrix)

        count=0

        for i in range(n):

            # rowSum = np.zeros(m, dtype=int)

            for j in range(i,n):

                if j==i:
                    rowSum = matrix[:,i]
                else:
                    rowSum=(rowSum+matrix[:,j])

                # rowSum = rowSum + matrix[:, j]

                time=self.sub_array_sum_1D(rowSum,target)

                print("i:{},j:{},rowSum:{}".format(i,j,rowSum))
                print("time:",time)

                count+=time

        return count

    def sub_array_sum_1D(self,nums,target):
        """
        和为 target 的子数组的个数
        
        :param nums: 
        :param target: 
        :return: 
        """
        nums=list(nums)

        n=len(nums)

        nums=[0]+nums

        prefix=[0 for __ in range(n+1)]

        dict_prefix={}
        dict_prefix[0]=[0]

        count=0

        for i in range(1,n+1):
            prefix[i]=prefix[i-1]+nums[i]

            diff=prefix[i]-target

            if diff in dict_prefix:
                count+=len(dict_prefix[diff])

            if prefix[i] not in dict_prefix:
                dict_prefix[ prefix[i] ]=[i]
            else:
                dict_prefix[prefix[i]].append(i)

        return count



class Test:
    def test_small_dataset(self, func):

        matrix = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        target = 0
        assert func(matrix,target) == 4

        matrix = [[1, -1], [-1, 1]]
        target = 0
        assert func(matrix,target) == 5

        matrix = [[0, 1, 1, 1, 0, 1],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 1, 0, 0, 1],
                  [1, 1, 0, 1, 1, 0],
                  [1, 0, 0, 1, 0, 0]]

        target = 0
        assert func(matrix, target) == 43

        # TODO: 边界条件
        assert func( [[904]],0) == 0

        assert func([[1]],1) == 1


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

    # print(sol.sub_array_sum_1D([1, -1, 0, 1, -1], 0))

    # matrix = [[0, 1, 0],
    #           [1, 1, 1],
    #           [0, 1, 0]]
    # target = 0

    matrix=[[0, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [1, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 0]]

    target=0

    print(sol.solve(matrix,target))



    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










