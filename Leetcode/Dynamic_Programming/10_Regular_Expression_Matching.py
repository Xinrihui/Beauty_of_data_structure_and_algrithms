#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

import numpy as np

class Solution:

    # isMatch
    def solve(self, s, p):
        """
        0 <= s.length <= 20
        0 <= p.length <= 30

        s contains only lowercase English letters.
        p contains only lowercase English letters, '.', and '*'.

        时间复杂度 

        :param s: 
        :param p:
        :return: 
        """

        s = ' ' + s  # ' abbb'
        p = ' ' + p  # ' ab*'

        L_s = len(s)  # 5
        L_p = len(p)  # 4

        dp = np.zeros((L_s, L_p), dtype='uint8')

        dp[0][0]=1 # s='' p='' 匹配

        #初始化：
        for j in range(1, L_p):

            if p[j]=='*':
                dp[0][j]=dp[0][j-2]


        for i in range(1, L_s):

            for j in range(1, L_p):

                # print(i,j)

                if s[i] == p[j] or p[j]=='.':

                    dp[i][j] = dp[i - 1][j - 1]

                elif p[j] == '*':

                        if p[j - 1] == s[i] or p[j - 1] == '.':  # p[j-1]==s[i]

                            # dp[i][j] = dp[i][j - 2] // 没有匹配的情况
                            # dp[i][j] = dp[i][j - 1] // 单个字符匹配的情况
                            # dp[i][j] = dp[i - 1][j] // 多个字符匹配的情况

                            # (1) X* == Null 即 p[j] 的前一个元素 X  匹配 0 次

                            L0 = dp[i][j - 2]

                            # (2) X* == X  即 p[j]的前一个元素 X 匹配 1 次

                            L1=dp[i][j - 1]

                            # (3) X* == XX  即 p[j]的前一个元素 X 匹配 大于1 次 (k=2,3..)
                            L2=dp[i-1][j]

                            dp[i][j] = (L0 | L1 | L2)

                        else:  # p 中'*' 的前一个 元素 不等于 s[i]

                            dp[i][j] = dp[i][j - 2]

                else:
                    dp[i][j] = 0

        print(dp)

        if dp[L_s-1][L_p-1] == 1:
            return True

        return False


class Solution_deprecated:

    # isMatch
    def solve(self, s, p):
        """
        0 <= s.length <= 20
        0 <= p.length <= 30
        
        s contains only lowercase English letters.
        p contains only lowercase English letters, '.', and '*'.

        时间复杂度 

        :param s: 
        :param p:
        :return: 
        """
        # 边界情况处理
        if s=='' and p=='':
            return True

        elif s=='' or p=='':
            return  False

        # 首位不等的特殊处理
        # if s[0]!=p[0]:


        s=' '+s # ' abbb'
        p=' '+p # ' ab*'

        L_s=len(s) # 5
        L_p=len(p) # 4

        dp=np.zeros((L_s,L_p),dtype=int)

        for i in range(1,L_s):

            for j in range(1,L_p):

                # print(i,j)

                if s[i]==p[j]:

                    dp[i][j]= dp[i-1][j-1]+1

                else: # s[i]!=p[j]

                    if p[j]=='.':
                        dp[i][j] = dp[i - 1][j - 1] + 1

                    elif  p[j]=='*':


                        if p[j-1]==s[i] or p[j-1]=='.' : # p[j-1]==s[i]


                            # (1) * == Null 即 p[j] 的前一个元素 匹配 0 次

                            L= dp[i][j-1]

                            # (2)  p[j]=='*' p[j]的前一个元素 匹配 k 次 (k=1,2,3..)

                            k=1
                            while L < i : # L 最大就取到 i

                                prev_L=L

                                L= dp[i-1][j-2+k]+1

                                if L == prev_L: # L没有增长 退出循环
                                    break

                                k+=1

                            dp[i][j]=L

                        else: # p 中'*' 的前一个 元素 不等于 s[i]

                            dp[i][j] = 0

                    else:
                        dp[i][j] = 0


        print(dp)

        if dp[-1][-1]== L_s-1:
            return True

        return  False

class Test:
    def test_small_dataset(self, func):

        assert func('abc','abc') == True

        assert func('abbb','ab*') == True

        assert func('ab', '.*') == True

        assert func('aab','c*a*b') == True

        assert func('mississippi', 'mis*is*p*.') == False

        assert func('ab', '.*c') == False

        assert func('aaa', 'aaaa') == False

        assert func('aaa', 'a*aa') == True

        # TODO: 边界条件
        assert func('', '.*') == True

        assert func('', 'a*') == True

        assert func('', '') == True


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

    # print(sol.solve('abbb','ab*'))

    # print(sol.solve('aab','c*a*b'))


    # print(sol.solve('ab', '.*'))

    # print(sol.solve('ab', '.*c'))

    # print(sol.solve('aaa', 'aaaa'))

    # print(sol.solve('aaa', 'a*aa'))

    # print(sol.solve('aaa', 'a*aa'))

    # print(sol.solve('mississippi', 'mis*is*p*.'))

    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










