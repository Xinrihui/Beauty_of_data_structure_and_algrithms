#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit


import numpy as np
class Solution:

    # isInterleave
    def solve(self, s1,s2,s3):
        """
        
        0 <= s1.length, s2.length <= 100
        0 <= s3.length <= 200

        时间复杂度 

        :param s: 
        :return: 
        """

        if len(s1)+len(s2)!=len(s3):
            return False

        L_s1=len(s1)
        L_s2=len(s2)
        L_s3=len(s3)

        # dp=np.zeros((L_s3+1,L_s1+1),dtype=bool)

        dp=[[False for __ in range(L_s1+1)] for __ in range(L_s3+1)]

        s1= ' '+s1
        s2 = ' ' + s2
        s3 = ' ' + s3

        # dp[0][0]=True
        # dp[0][1]=True

        for i in range(L_s1+1):
            dp[0][i]=True


        for k in range(1,L_s3+1):
            for i in range(0,k+1):

                j=k-i
                if i<=L_s1 and j <= L_s2: # j 的长度要 在范围内

                    # print('k:{},i:{},j:{}'.format(k,i,j))

                    if i==0:# 只能 取s2

                        dp[k][i]=(dp[k-1][i] and s3[k]==s2[j])

                    elif i==k: # 只能取 s1

                        # if s3[k]==s1[i]:
                            dp[k][i]=(dp[k-1][i-1] and s3[k]==s1[i])

                    else:

                        case1=(dp[k-1][i] and s3[k]==s2[j])

                        case2=(dp[k-1][i-1] and s3[k]==s1[i])

                        dp[k][i]=case1 or case2


        # print(dp)

        return dp[L_s3][L_s1]




class Test:
    def test_small_dataset(self, func):

        assert func("aabcc", "dbbca", "aadbbcbcac") == True

        assert func("aabcc", "dbbca","aadbbbaccc") == False

        assert func("aabcc","dbbca","aadbbbcacc") == True

        assert func("a", "b", "ab") == True

        assert func("ab", "c", "abc") == True

        assert func("a", "bc", "abc") == True

        assert func("a", "cb", "abc") == False

        # TODO: 边界条件
        # assert func(None) == None

        assert func('a', '', 'a') == True

        assert func('', 'b', 'b') == True

        assert func('', 'a', 'b') == False

        assert func('','','') == True

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

    # print(sol.solve("aabcc","dbbca","aadbbbcacc"))

    # print(sol.solve("aabcc", "dbbca", "aadbbcbcac"))

    # print(sol.solve("aab", "dbb", "aadbbb"))


    print(sol.solve("a", "b", "ab"))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










