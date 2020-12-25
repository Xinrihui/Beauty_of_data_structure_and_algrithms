#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

class Solution:

    # numDecodings(self, s: str) :
    def solve(self, s):
        """

        时间复杂度 
        
        :param s: 
        :return: 
        """
        L_s=len(s)

        s = ' ' + s

        # dp=np.zeros((L_s+1),dtype=int) # 方便调试
        dp = [0 for __ in range(L_s+1)]

        # D_list = np.zeros((L_s + 1), dtype=int)
        D_list = [0 for __ in range(L_s+1)]

        if s[1]=='0':
            return 0
        else:
            D_list[1]=0
            dp[1]=1

        split_set=set(range(1,10))|set(range(27,100))-{i*10 for i in range(3,10)}
        merge_set={10,20}
        merge_split_set=set(range(10,27))-merge_set
        no_merge_no_split_set={i*10 for i in range(3,10)}|{0}

        print('split_set:',split_set)
        print('merge_set:', merge_set)
        print('merge_split_set:', merge_split_set)
        print('no_merge_no_split_set:', no_merge_no_split_set)

        for i in range(2,L_s+1):

            if int(s[i-1:i+1]) in split_set:
                D_list[i]=0

            elif int(s[i-1:i+1]) in merge_set:
                D_list[i] = 1

            elif int(s[i - 1:i + 1]) in merge_split_set:
                D_list[i] = 2

            elif int(s[i - 1:i + 1]) in no_merge_no_split_set:
                D_list[i] = -1


            if  D_list[i] == -1:
                return 0

            if D_list[i-1]==0 and D_list[i]==0:
                dp[i]=dp[i-1]

            elif D_list[i-1]==0 and D_list[i]==1:
                dp[i]=dp[i-1]

            elif D_list[i-1]==0 and D_list[i]==2:
                dp[i]=dp[i-1]*2

            elif D_list[i-1]==1 and D_list[i]==0:
                dp[i]=dp[i-1]

            elif D_list[i-1]==2 and D_list[i]==0:
                dp[i]=dp[i-1]

            elif D_list[i-1]==2 and D_list[i]==1:
                dp[i]=dp[i-2]

            elif D_list[i-1]==2 and D_list[i]==2:
                dp[i]=dp[i-1]+dp[i-2]

        print('D_list:',D_list)
        print('dp:',dp)

        return dp[-1]



class Test:
    def test_small_dataset(self, func):

        assert func("2234") == 3

        assert func("2213") == 5

        assert func("231023") == 4

        assert func("23014") == 0

        assert func("10") == 1

        assert func("2101") == 1

        assert func("123123") == 9

        # TODO: 边界条件
        assert func("0") == 0

        assert func('1') == 1

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

    # print(sol.solve("10"))

    print(sol.solve("123123"))

    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










