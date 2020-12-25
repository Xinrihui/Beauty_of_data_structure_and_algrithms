#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class Solution:

    # countPalindromicSubsequences
    def solve(self, s: str) -> int:

        n = len(s)

        s = ' ' + s

        # count=np.zeros((n+1,n+1),dtype=int )
        count = [[0 for __ in range(n + 1)] for __ in range(n + 1)]

        for l in range(1, n + 1):
            for i in range(1, n - l + 2):

                j = i + l - 1

                # print('l:{},i:{},j:{}'.format(l,i,j))

                if l == 1:
                    count[l][i] = 1

                elif l >= 2:

                    if s[i] == s[j]:

                        left = i + 1
                        right = j - 1

                        while left <= j and s[left] != s[i]: # 从 i 的右侧 开始, 找到 第1个s[i]
                            left += 1

                        while right >= i and s[right] != s[i]: # 从 j 的左侧 开始, 找到 第1个s[i]
                            right -= 1

                        # print('left:{},right:{}'.format(left,right))

                        if left == right : # s[i+1:j] 中只有1个 s[i]
                            count[l][i] = 2 * count[l - 2][i + 1] + 1

                        elif left < right: # s[i+1:j] 中 找到2个

                            count[l][i] = 2 * count[l - 2][i + 1] - count[right - left - 1][left + 1]

                        elif left > right: #  s[i+1:j] 中 没有  s[i]

                            count[l][i] = 2 * count[l - 2][i + 1] + 2

                    else:

                        count[l][i] = count[l - 1][i] + count[l - 1][i + 1] - count[l - 2][i + 1]

                # print('count:',count[l][i])

        # print(count)

        return count[n][1] % (10**9+7)


class Test:
    def test_small_dataset(self, func):

        assert func('bccb') == 6

        assert func("bbbb") == 4

        assert func("bbcabb") == 10

        assert func("bcbcb") == 9


        assert func('abcdabcdabcdabcdabcdabcdabcdabcddcbadcbadcbadcbadcbadcbadcbadcba')==104860361

        # TODO: 边界条件
        # assert func(None) == None

        assert func('a') == 1


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

    print(sol.solve("bccb"))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










