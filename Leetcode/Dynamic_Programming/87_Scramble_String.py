#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit


import numpy as np

class Solution:

    # isScramble
    def solve(self, s1,s2):
        """

        时间复杂度: O(N^3) N=len(s1)

        :param s1: 
        :param s2:
        :return: 
        """

        L_s=len(s1)

        # dp=np.zeros((L_s+1,L_s,L_s),dtype=bool)

        dp=[[[False for __ in range(L_s)] for __ in range(L_s)] for __ in range(L_s+1)]

        for l in range(1,L_s+1):# 子串的长度

            for p in range(L_s-l+1): # s1 子串的开始位置

                for q in range(L_s-l+1):  # s2 子串的开始位置

                    #判断 s1 子串 和 s2 子串 是否可以 通过旋转 进行转化
                    if l==1:
                        if s1[p]==s2[q]:
                            dp[l][p][q]=True

                    # elif l>=2 and l<=3:
                    #     if set(s1[p:p+l])==set(s2[q:q+l]): # TODO: s1=aac s2=acc 显然不可以 通过旋转 进行转化
                    #         dp[l][p][q] = True
                    # elif l>=4:

                    else:  # l=2,3,...

                        # 判断 s1[p:p+l] 和 s2[q:q+l] 可以 通过旋转 进行转化 的标志

                        # 对 s1[p:p+l] 进行切分
                        for i in range(1,l):

                            j_s=i
                            j_e=l-i

                            case1= dp[i][p][q] and dp[l-i][p+i][q+j_s]
                            case2= dp[i][p][q+j_e] and dp[l-i][p+i][q]

                            if case1 or case2:
                                dp[l][p][q] = True

                                break

        # print(dp)

        return dp[L_s][0][0]




class Test:
    def test_small_dataset(self, func):


        assert func('caba','baac') == True

        assert func('great','rgeat') == True

        assert func('great', 'rgtae') == True

        assert func('abcde', 'caebd') == False

        assert func('aaccd','acaad') == False


        # TODO: 边界条件
        # assert func(None) == None

        assert func('a','a') == True

        assert func('a', 'b') == False

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

    # print(sol.solve('caba','baac'))

    print(sol.solve('abcde', 'caebd'))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










