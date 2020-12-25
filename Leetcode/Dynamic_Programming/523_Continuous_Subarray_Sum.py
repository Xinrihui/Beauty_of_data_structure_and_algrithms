#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

class Solution:

    #  checkSubarraySum(self, nums: List[int], k: int) -> bool:

    def solve(self, nums,k):
        """

        时间复杂度 

        :param s: 
        :return: 
        """
        n=len(nums)

        nums=[0]+nums # 数组 前面加上0 ,生成新的数组, 不改变原来的 nums

        prefix=[0 for __ in range(n+1)] #

        prefix_modK=[0 for __ in range(n+1)]

        dict_prefix_modK={}
        dict_prefix_modK[0]=[0]

        # count=0

        for i in range(1,n+1):

            prefix[i]=prefix[i-1]+nums[i]

            if k!=0:
                prefix_modK[i]= (prefix[i] % k)
            else:
                prefix_modK[i] =prefix[i]

            if prefix_modK[i] in dict_prefix_modK:

                idx_list=dict_prefix_modK[ prefix_modK[i] ]

                # if i - idx_list[-1]>1:
                #     count+=1

                idx_list.append(i)

            else:
                dict_prefix_modK[prefix_modK[i]]=[i]

        # print(prefix)
        # print(prefix_modK)
        #
        # print(dict_prefix_modK)

        flag=False

        for v_list in dict_prefix_modK.values():

            if len(v_list)>1:
                if v_list[-1]-v_list[0]>1:
                    flag=True
                    break

        return flag



class Test:
    def test_small_dataset(self, func):

        assert func([23, 2, 4, 6, 7], 6) == True


        assert func([23, 2, 6, 4, 7],0) == False

        assert func([0, 0],0) ==True
        #
        # assert func("cbbd") == 'bb'
        #
        # assert func("cbbc") == 'cbbc'
        #
        # assert func("abcd") == 'a'

        # TODO: 边界条件
        assert func([1],1) == False

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

    sol = Solution()

    # IDE 测试 阶段：


    # print(sol.solve([23, 2, 4, 6, 7], 6))

    print(sol.solve([0, 0, 0, 0, 0],0))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










