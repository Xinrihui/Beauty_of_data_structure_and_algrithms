#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

import numpy as np

class Solution:

    # longestValidParentheses
    def solve(self, s):
        """
        M2: 动态规划
        
        :param s: 
        :return: 
        """

        s=' '+s

        print(s)

        L_s=len(s)

        dp=np.zeros(L_s,dtype=int)

        for i in range(1,L_s):

            if s[i]=='(':
                dp[i]=0

            elif s[i]==')'and s[i-1]=='(':

                dp[i]=dp[i-2]+2

            elif s[i] == ')' and s[i - 1] == ')':

                if s[i-dp[i-1]-1]=='(':
                    dp[i]=(dp[i-1]+2)+(dp[i-dp[i-1]-1-1])

        print(dp)

        res=max(dp)

        return res

    # longestValidParentheses
    def solve_M1(self, s):
        """

        利用 栈的性质 

        0 <= s.length <= 3 * 10^4

        
        时间复杂度:

        :param s: 
        :return: 
        """

        if len(s)==0:
            return 0

        L_s=len(s)

        p=np.zeros(L_s,dtype=int)

        stack=[]

        for i in range(L_s):

            if s[i]=='(':

                stack.append(s[i])
                p[i]= -1

            elif  s[i]==')' and len(stack)==0:

                p[i] = 0

            elif s[i]==')'  and len(stack)>0:

                stack.pop()

                p[i] = 1

                j=i-1

                while p[j]!=-1 and j>=0:

                    j-=1

                if j>=0:
                    p[j] = 1

        print(p)

        max_L=float('-inf')
        L=0

        for i in range(L_s):

            if p[i] ==1:

                L+=1
                if L > max_L:
                    max_L=L

            else:
                L=0


        return max(0,max_L)

    def solve_M1_opt(self,s):
        """
        利用 栈的性质 
        
        优化： 在栈中 放入括号元素的标号
        
        :param s: 
        :return: 
        """
        stack = []
        match = [0 for i in range(0, len(s))]
        for i, c in enumerate(s):
            if c == '(':
                stack.append(i)
            elif c == ')' and len(stack) != 0:
                match[i] = 1
                match[stack[-1]] = 1
                stack.pop()

        print(match)

        ans, temp = 0, 0
        for i, c in enumerate(match):
            if match[i]:
                temp = temp + 1
                ans = max(ans, temp)
            else:
                temp = 0
        return ans



class Test:
    def test_small_dataset(self, func):


        assert func(")()())") == 4

        assert func(")((()))(") == 6

        assert func(")((()())") == 6

        assert func("(()") == 2

        assert func(")())()") == 2

        assert func(")(") == 0

        assert func("()(())") == 6

        assert func("())(())") == 4


        assert func("()(()") == 2

        # TODO: 边界条件
        # assert func(None) == None

        assert func('') == 0

        assert func('(') == 0
        assert func(')') == 0

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

    # print(sol.solve(")()())"))
    #
    # print(sol.solve(")((()))("))


    # print(sol.solve("()(())"))
    #
    # print(sol.solve("()(()"))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










