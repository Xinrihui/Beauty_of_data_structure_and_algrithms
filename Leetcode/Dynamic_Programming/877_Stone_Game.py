#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

class Solution:

    #  stoneGame(self, piles: List[int]) -> bool:
    def solve(self, p):
        """

        时间复杂度 

        :param piles: 
        :return: 
        """
        L_p=len(p)

        p=[0]+p # 数组 前面加上0 ,生成新的数组, 不改变原来的 nums

        # first=np.zeros((L_p+1,L_p+1),dtype=int) # 先手

        first =[[0 for __ in range(L_p+1)] for __ in range(L_p+1)]

        # second = np.zeros((L_p + 1, L_p + 1), dtype=int) # 后手
        second = [[0 for __ in range(L_p + 1)] for __ in range(L_p + 1)]

        l=1
        for i in range(1, L_p - l + 1+1):
            first[l][i]=p[i]


        for l in range(2,L_p+1):
            for i in range(1,L_p-l+1+1):
                j=i+l-1

                # 1.先手先选
                left=p[i]+second[l-1][i+1] # 最左边的元素
                right=p[j]+second[l-1][i]  # 最右边的元素


                if left>right:
                    first[l][i] = left
                    # 2. 后手依据 先手的 选择 再选
                    second[l][i]=first[l-1][i+1]
                else:
                    first[l][i] = right

                    second[l][i] =first[l-1][i]

        # print(first)
        # print(second)

        flag=None

        if first[L_p][1] > second[L_p][1]:
            flag=True
        elif first[L_p][1] < second[L_p][1]:
            flag=False

        return flag


class Test:
    def test_small_dataset(self, func):

        assert func([3,9,1,2]) == True

        assert func([5,3,4,5]) == True

        # assert func("cbbd") == 'bb'
        #
        # assert func("cbbc") == 'cbbc'
        #
        # assert func("abcd") == 'a'

        # TODO: 边界条件
        # assert func(None) == None
        #
        # assert func('') == ''

    def read_test_case_fromFile(self,dir):


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
        K, l1 = self.read_test_case_fromFile(dir)

        start = timeit.default_timer()
        print('run large dataset:{} '.format(dir))
        func(K, l1)  # 12.047259273 s
        end = timeit.default_timer()
        print('time: ', end - start, 's')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    print(sol.solve([3,9,1,2]))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










