#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit


import numpy as np
class Solution:

    #longestPalindrome
    def solve(self, s):
        """
        动态规划解 最长回文子串问题
        
        时间复杂度 O(n^2)
        
        :param s: 
        :return: 
        """

        # not s == s is None or s==''
        if not s:
            return s

        L=len(s)

        states= np.zeros((L+1,L),dtype='int')

        # 1. 遍历不同长度的子串, 找出其中的回文子串

        for l in range(1,L+1): # 子串的长度  l=1,2,3,..

            for start in range(L-l+1): # 子串的开始位置

                if l==1:
                    states[l][start]=1 # 子串的长度为1时 回文是它自己

                else: # l>1

                    end=start+l-1 # 子串结束位置的指针

                    if s[start]==s[end]: # 子串 的首尾 字符相同


                        if l==2: # 子串的长度是2, 首尾字符相同, 肯定是回文
                            states[l][start]=2
                            continue

                        # 子串的长度为 偶数 或者 奇数
                        if states[l-2][start+1]!=0: # 里面一层的子串是回文

                            states[l][start]=states[l-2][start+1]+2

                        else:
                            states[l][start]=0


        # 2. 回文子串中 找最长的
        # print(states)

        max_ele= np.max(states)

        max_idx_list= np.where(states==max_ele) #(array([3, 3], dtype=int64), array([0, 1], dtype=int64))
                                        #  最大值在 第3行第0列 和 第3行第3列

        # print(max_ele)
        # print(max_idx_list)

        first_idx= (max_idx_list[0][0],max_idx_list[1][0]) # 取其中第一个 回文子串

        l=states[first_idx[0]][first_idx[1]] # 回文子串 的长度

        return  s[first_idx[1]:first_idx[1]+l]


class Test:

    def test_small_dataset(self, func):

        assert func("babad") == 'bab'

        assert func("babab") == 'babab'

        assert func("cbbd") == 'bb'

        assert func("cbbc") == 'cbbc'

        assert func("abcd") == 'a'

        #TODO: 边界条件
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

    # print(sol.solve("babad"))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










