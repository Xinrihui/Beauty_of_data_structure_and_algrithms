#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit


import numpy as np

class Solution:

    # calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
    def solve(self, M):
        """
        
        反向 DP 

        时间复杂度 

        :param M: 
        :return: 
        """
        m,n=len(M),len(M[0])

        # dp=np.zeros((m,n),dtype=int) # 方便调试

        dp = [[0 for __ in range(n)] for __ in range(m)]

        # 初始化

        dp[m-1][n-1]=M[m-1][n-1]

        i=m-1
        for j in range(n-2,-1,-1):
            dp[i][j]= min( M[i][j] ,dp[i][j+1] + M[i][j])

        j = n - 1
        for i in range(m - 2, -1, -1):
            dp[i][j] =min( M[i][j] , dp[i+1][j] + M[i][j])


        for i in range(m - 2, -1, -1):
            for j in range(n - 2, -1, -1):

                dp[i][j]= min(M[i][j],max(dp[i][j+1],dp[i+1][j]) + M[i][j])


        # print(dp)

        min_HP=1 # min_HP >=1

        if dp[0][0]<0:
            min_HP=0-dp[0][0] +1

        return min_HP


class Solution_Deprecated:

    # calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
    def solve(self, M):
        """

        时间复杂度 

        :param M: 
        :return: 
        """
        m,n=len(M),len(M[0])

        hp = np.zeros((m, n), dtype=int)
        dp=np.zeros((m,n),dtype=int)

        # dp=[[[False for __ in range(L_s)] for __ in range(L_s)] for __ in range(L_s+1)] # 提高效率

        # 初始化
        hp[0][0]=M[0][0]


        dp[0][0] = M[0][0] # dp[i][j] <=0


        i=0 # 只能往下走
        for j in range(1, n):
            hp[i][j]=hp[i][j-1]+M[i][j]
        j = 0
        for i in range(1, m):
            hp[i][j] = hp[i-1][j] + M[i][j]


        i = 0 # 只能往右走
        for j in range(1, n):
            dp[i][j] = min( hp[i][j - 1] + M[i][j], dp[i][j-1])
        j = 0
        for i in range(1, m):
            dp[i][j] = min( hp[i-1][j] + M[i][j], dp[i-1][j])

        # print(hp)
        # print(dp)

        for i in range(1,m):
            for j in range(1, n):

                hp[i][j]=max(hp[i][j-1],hp[i-1][j])+M[i][j]
                dp[i][j] = max(min( hp[i][j - 1] + M[i][j], dp[i][j-1]),min(hp[i - 1][j] + M[i][j], dp[i - 1][j]))

                # right=min( hp[i][j - 1] + M[i][j], dp[i][j-1])
                # down=min(hp[i - 1][j] + M[i][j], dp[i - 1][j])
                #
                # if down>= right: # 往下走
                #
                #     dp[i][j]=down
                #     hp[i][j] =hp[i - 1][j] + M[i][j]
                #
                # else: # 往右走
                #     dp[i][j] = right
                #     hp[i][j] = hp[i][j-1] + M[i][j]


        print(hp)
        print(dp)

        min_HP=1 # min_HP >=1

        if dp[-1][-1]<0:
            min_HP=0-dp[-1][-1] +1

        return min_HP

class Test:
    def test_small_dataset(self, func):

        assert func([[-2,-3,3],[-5,-10,1],[10,30,-5]]) == 7

        assert func([[100]]) == 1

        M = [[1, -4, 5, -99],
             [2, -2, -2, -1]]

        assert func(M) == 3

        M=[[1, -3, 3],
           [0, -2, 0],
           [-3, -3, -3]]

        assert func(M) == 3

        # assert func("cbbd") == 'bb'
        #
        # assert func("cbbc") == 'cbbc'
        #
        # assert func("abcd") == 'a'

        # TODO: 边界条件
        # assert func(None) == None
        #
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

    print(sol.solve([[-2,-3,3],[-5,-10,1],[10,30,-5]]))

    # M=[[1, -4, 5, -99],
    #  [2, -2, -2, -1]]

    # print(sol.solve(M))

    M=[[1, -3, 3], [0, -2, 0], [-3, -3, -3]]
    # print(sol.solve(M))

    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










