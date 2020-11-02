#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit


import numpy as np


class Solution:

    def solve(self, obstacleGrid):

        m, n = len(obstacleGrid), len(obstacleGrid[0])

        if m == 0 or n == 0:
            return 1

        dp = [[0 for __ in range(n)] for __ in range(m)]

        # 初始化 边界情况
        if obstacleGrid[0][0] == 1:
            return 0

        dp[0][0] = 1

        i = 0
        for j in range(1, n):
            if obstacleGrid[i][j] == 0:  # 没有障碍物
                dp[i][j] = dp[i][j - 1]
            else:
                dp[i][j] = 0

        j = 0
        for i in range(1, m):
            if obstacleGrid[i][j] == 0: # 没有障碍物
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = 0


        for i in range(1, m):
            for j in range(1, n):

                if obstacleGrid[i][j] == 1:  # 有障碍物
                    dp[i][j] = 0

                else:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        print(dp)

        return dp[-1][-1]


class Test:
    def test_small_dataset(self, func):


        # TODO: 边界条件

        assert func([[]]) == 1

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
    obstacleGrid=[[0,0,0],[0,1,0],[0,0,0]]
    print(sol.solve(obstacleGrid))


    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










