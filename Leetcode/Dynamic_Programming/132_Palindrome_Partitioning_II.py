#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit


import numpy as np

class Solution:

    # def minCut(self, s: str) -> int:
    def solve(self, s):
        """

        时间复杂度 

        :param s: 
        :return: 
        """
        L_s=len(s)


        # 1. 预处理
        states=self.getPalindrome(s)

        # print(states)

        # s = ' ' + s

        dp=[0 for __ in range(L_s+1)]

        for i in range(1,L_s+1):

            min_num=float('inf')

            for k in range(i):

                # print('i:{} k:{}'.format(i,k))

                if states[i-k][k+1]==True:
                    num=dp[k]+1
                else:
                    num=float('inf')

                min_num=min(num,min_num)

            dp[i]=min_num

        # print(dp)

        n=dp[-1]# 最少回文串 个数
        cut=n-1 # 最小切分个数

        return cut

    def getPalindrome(self, s):
        """
        记录所有的 子串是否为 回文

        时间复杂度 O(n^2)

        :param s: 
        :return: 
        """

        # not s == s is None or s==''
        if not s:
            return s

        L_s = len(s)

        s=' '+s # TODO: 检查 s 是否之前已经 在前面添加了 ' '

        # states = np.zeros((L_s+1, L_s+1), dtype=bool)

        states = [[False for __ in range(L_s+1)] for __ in range(L_s+1)]


        # 1. 遍历不同长度的子串, 找出其中的回文子串

        for l in range(1, L_s + 1):  # 子串的长度  l=1,2,3,..

            for start in range(1,L_s - l + 2):  # 子串的开始位置

                # print('l:{} , start:{}'.format(l,start))

                if l == 1:
                    states[l][start] = True  # 子串的长度为1时 回文是它自己

                elif l==2 and s[start] == s[start+1]:
                    states[l][start] = True

                elif l>=3 and  s[start] == s[start+l-1]:

                    states[l][start]= states[l-2][start+1]

        return states

class Test:
    def test_small_dataset(self, func):

        assert func("aabaabac") == 2

        assert func("aab") == 1

        assert func("a") == 0

        assert func("ab") == 1

        assert func("bb") == 0

        assert func("abcd") == 3

        assert func("caabaabb") == 2

        # TODO: 边界条件
        # assert func(None) == None

        # assert func('') == ''

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

    # print(sol.getPalindrome('caabaabb'))

    # print(sol.solve("aabaabac"))

    # print(sol.solve("abcd"))

    # print(sol.solve("aa"))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










