#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # longestPalindrome
    def solve(self, s):
        """


        时间复杂度 

        :param s: 
        :return: 
        """
        # L_s=len(s)

        # s = ' ' + s
        # nums=[0]+nums # 数组 前面加上0 ,生成新的数组, 不改变原来的 nums

        # dp=np.zeros((L_s+1,L_s,L_s),dtype=bool) # 方便调试
        # dp=[[[False for __ in range(L_s)] for __ in range(L_s)] for __ in range(L_s+1)] # 提高效率

        # for i in range(1,L_s+1):


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

    def test_large_dataset(self, func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题

        Limits


        :param func: 
        :return: 
        """

        N = int(2 * pow(10, 4))
        max_v = int(pow(10, 9))

        l = np.random.randint(max_v, size=N)
        l1 = list(l)

        start = timeit.default_timer()
        print('run large dataset: ')
        func()
        end = timeit.default_timer()
        print('time: ', end - start, 's')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    print(sol.solve("babad"))


    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










