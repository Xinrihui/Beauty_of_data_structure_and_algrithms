#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit


import numpy as np

from collections import *

class Solution:

    # numDistinct(self, s: str, t: str)
    def solve(self, s, t) :
        """


        时间复杂度 

        :param s: 
        :return: 
        """
        L_s=len(s)
        L_t=len(t)

        s=' '+s
        t=' '+t

        index_list=[[] for i in range(L_t+1)] # s 中每一个字符 的标号 列表

        dp=[[] for i in range(L_t+1)]

        index_list[0]=[0] # 初始化
        dp[0] = [1]

        # hash_char_t_index={ t[i]:i  for i in range(1,L_t+1)}

        hash_t_char_index = {} # t 中 字符 和标号的关系

        # t 中的字符 可能重复
        for i in range(1,L_t+1):

            if t[i] not in hash_t_char_index: # 第一次看到 t[i]

                hash_t_char_index[t[i]]=[i]

            else:

                hash_t_char_index[t[i]].append(i)

        # print(hash_char_t_index) # {'r': [1], 'a': [2], 'b': [3, 4], 'i': [5], 't': [6]}

        for i in range(1,L_s+1):

            if s[i] in hash_t_char_index: # t 中有的 字符 s 中可能没有

                for ele in hash_t_char_index[s[i]]:

                    index_list[ele].append(i)
                    dp[ele].append(0)


        # print(index_list)

        for level in range(1,L_t + 1):

            for i in range(len(index_list[level])): # 当前层的 元素

                for j in range(len(index_list[level-1])): # 前一层的元素

                    if index_list[level][i]>index_list[level-1][j]: # 当前层 元素 > 前一层元素

                        dp[level][i]+=dp[level-1][j]

        # print(dp)

        return sum(dp[L_t])


class Test:
    def test_small_dataset(self, func):


        assert func("babgbag", "bag") == 5

        assert func("rabbbit", "rabbit") == 3

        assert func("b","a") == 0

        assert func("ba","ab") == 0

        assert func("dabc", "ab") == 1

        assert func("dabc", "ac") == 1

        assert func("dabc", "ca") == 0


        # TODO: 边界条件
        # assert func(None) == None
        #
        # assert func('','') == 0

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

    # print(sol.solve("babgbag", "bag"))
    #
    # print(sol.solve("rabbbit", "rabbit"))

    # print(sol.solve("a", "b"))

    print(sol.solve( "dabc", "ab"))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










