#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

import numpy as np

class Solution:

    # isMatch(self, s: str, p: str) -> bool:
    def solve(self, s, p):
        """

        时间复杂度  

        :param s: 
        :return: 
        """

        p=self.__pre_process(p) # 多个 * 和1个 * 效果相同, 但是对运行时间没有本质影响

        L_s = len(s)
        L_p = len(p)

        s = ' ' + s
        p = ' ' + p

        # dp=np.zeros((L_s+1,L_p+1),dtype=bool) # 方便调试

        dp = [[False for __ in range(L_p + 1)] for __ in range(L_s + 1)]  # 提高效率

        # 初始化
        # (1) s='' p='' 匹配
        dp[0][0] = True

        # (2) '*' 为首字符 需要特殊处理
        i = 0
        for j in range(1, L_p + 1):
            if p[j] == '*':  #  p 中从 头开始 连续为 '*'
                dp[i][j] = True
            else:
                break

        for i in range(1, L_s + 1):
            for j in range(1, L_p + 1):

                if s[i] == p[j] or p[j] == '?':
                    dp[i][j] = dp[i - 1][j - 1]

                elif p[j] == '*' :

                    dp[i][j] = (dp[i][j - 1] or dp[i - 1][j])

        # print(dp)

        return dp[L_s][L_p]

    def __pre_process(self,p_orgin):
        """
        删除p 中多余的 * 
        :param p_orgin: 
        :return: 
        """
        p = ''

        for i in range(len(p_orgin)):
            if p_orgin[i] != '*':
                p += p_orgin[i]
            else:
                if len(p) == 0 or p[-1] != '*':
                    p += p_orgin[i]

        return p

    # isMatch(self, s: str, p: str) -> bool:
    def solve_Deprecated(self, s ,p):
        """
        超时 TLE 
        
        时间复杂度  

        :param s: 
        :return: 
        """


        L_s=len(s)
        L_p = len(p)

        s = ' ' + s
        p = ' ' + p

        # dp=np.zeros((L_s+1,L_p+1),dtype=bool) # 方便调试

        dp=[[False for __ in range(L_p+1)] for __ in range(L_s+1)]  # 提高效率

        # 初始化
        # (1) s='' p='' 匹配
        dp[0][0]=True

        # (2) '*' 为首字符 需要特殊处理
        i=0
        for j in range(1,L_p + 1):
            if p[j]=='*': # 找到连续为 '*'
                dp[i][j] = True
            else:
                break

        # if L_p>0 and  p[1]=='*': #  '*' 为首字符 需要特殊处理
        #     dp[0][1] =True

        for i in range(1,L_s+1):
            for j in range(1, L_p + 1):

                if s[i]==p[j] or p[j]=='?':
                    dp[i][j]=dp[i-1][j-1]

                elif  p[j]=='*':

                    ii=i
                    while ii>=0: # TODO：超时 原因，需要优化

                        if dp[ii][j-1]==True:
                            dp[i][j] = dp[ii][j-1]
                            break
                        ii-=1

        # print(dp)

        return dp[L_s][L_p]


class Test:
    def test_small_dataset(self, func):

        s = "acdcb"
        p = "a*c?b"
        assert func(s,p) == False

        s = "adceb"
        p = "*a*b"
        assert func(s, p) == True

        s = "aa"
        p = "a"
        assert func(s, p) == False

        s = "aa"
        p = "*"
        assert func(s, p) == True

        s = "cb"
        p = "?a"
        assert func(s, p) == False

        s = "zacabz"
        p = "*a?b*"
        assert func(s, p) == False


        s = "a"
        p = "*?"
        assert func(s, p) == True

        s = "abcd"
        p = "**"
        assert func(s, p) == True

        s = "abcd"
        p = "a**d"
        assert func(s, p) == True

        s = "abcd"
        p = "**c"
        assert func(s, p) == False


        # TODO: 边界条件
        s = ""
        p = "*"
        assert func(s, p) == True

        s = ""
        p = ""
        assert func(s, p) == True

        s = "a"
        p = ""
        assert func(s, p) == False

        s = ""
        p = "a"
        assert func(s, p) == False

        s =""
        p = "?"
        assert func(s, p) == False

        s=""
        p="******"

        assert func(s, p) == True


    def test_large_dataset(self, func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题

        Limits


        :param func: 
        :return: 
        """

        # N = int(2 * pow(10, 4))
        # max_v = int(pow(10, 9))
        #
        # l = np.random.randint(max_v, size=N)
        # l1 = list(l)

        s="abbabaaabbabbaababbabbbbbabbbabbbabaaaaababababbbabababaabbababaabbbbbbaaaabababbbaabbbbaabbbbababababbaabbaababaabbbababababbbbaaabbbbbabaaaabbababbbbaababaabbababbbbbababbbabaaaaaaaabbbbbaabaaababaaaabb"
        p="**aa*****ba*a*bb**aa*ab****a*aaaaaa***a*aaaa**bbabb*b*b**aaaaaaaaa*a********ba*bbb***a*ba*bb*bb**a*b*bb"

        start = timeit.default_timer()
        print('run large dataset: ')
        func(s,p)
        end = timeit.default_timer()
        print('time: ', end - start, 's')

        s = "aaaaaaabababbababaaaabbbbbabababbbbbbaabbbbbbabbbabbbbaaaaaabababbabbaaabbbbababbbaaaaaaaababaaababababababbbbaabaabababbbbabbbbaabbaababbbaaabbabbbabaaabababaaaabaababaaaaabaaabbbabbaabbbbabbabbaaaaaaaa"
        p = "a*a***a*bbb*abb**babbba****ba*aa*a**a*aba*ba***b*a*ab**bb**b***b*b**a*aabbba*ab*a*******bba*a*a******"
        start = timeit.default_timer()
        print('run large dataset: ')
        func(s,p)
        end = timeit.default_timer()
        print('time: ', end - start, 's')

        s="bbbbaaaaabaabbbbaabaaabaabbababbbaaabbababbbabaabaabaabababaaabaaaabbaabbaabbaaaaabbabbbbaaaababbaaaabbabbbaabaaabbaabaabaaababbabbaababaababbbbbaabbabbabbbbaabbaaababbabaaabbbbbbbbaababbbbbbabbaabaaa"
        p="b*a**b***abaabaaaba*abaaaaabaabb*bbb*aa*ab*a**b**b*a**a**a*abbb***bb*b*****baababaa**ab*aa*bbaba**bb*b*"
        start = timeit.default_timer()
        print('run large dataset: ')
        func(s,p)
        end = timeit.default_timer()
        print('time: ', end - start, 's')

        s="bbaaabbababbbbbabababaaabbabbaabbbbbaaabbbaaababbbbababbbbabbbababaabababbbbbbbababaaababababbbbaabbaaaabbbaaabbbaababbbbababbbbbabbabbaabaabbaabaabbbabaabbbbbabababbabaabbababbabbbbabbbbaaaaaaaabbaab"
        p="a**abaaa*b*aa*ba*****b*a*bb**bbab*a*aa**b***ba*a*aabb*bab**aa*ab*b**b*b*aabba******bbbb*aa*a****abb***b*"

        start = timeit.default_timer()
        print('run large dataset: ')
        func(s,p)
        end = timeit.default_timer()
        print('time: ', end - start, 's')

        s="ababbaabababbaaaaabbaaaaaababaabaaabaabbbbaabbabbaaaaaaababaabbabbabbbababbbaabbabbaaaababbaaabbbbbaabbaabbabaababbabbbbbaababbbaababbaaababaaaaabababbbbbbbaaaaababaabababbabbaabababababaabaabaabaaabbaabb"
        p="aa****baa***b*ab*aa*a***a****b**b*b***a*b*b*a****a****b*a***ab**aab**b*a***ab**a***baabaa*aaa**b*aabb*"
        start = timeit.default_timer()
        print('run large dataset: ')
        func(s, p)
        end = timeit.default_timer()
        print('time: ', end - start, 's')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    # s = "acdcb"
    # p = "a*c?b"

    # s = "aa"
    # p = "*"

    s = "adceb"
    p = "*a*b"


    # s ="zacabz"
    # p ="*a?b*"

    print(sol.solve(s,p))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    test.test_large_dataset(sol.solve)










