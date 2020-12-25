#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class Solution:

    #  def profitableSchemes(self, G: int, P: int, group: List[int], profit: List[int]) -> int:
    def solve_3D_opt(self, G: int, P: int, group, profit):
        """
        01 背包问题

        3维动态规划 (优化)
        状态压缩 

        G=10^2
        n=10^2
        P=10^2

        时间复杂度 O(GnN)=10^6

        :return: 

        """

        n = len(group)

        group = [0] + group
        profit = [0] + profit

        # dp = np.zeros((G + 1, P + 1), dtype=int)

        dp = [[0 for __ in range(P + 1)] for __ in range(G + 1)]

        dp[0][0] = 1

        for i in range(1, n + 1):
            for w in range(G, group[i] - 1, -1):
                for v in range(P, -1, -1):

                    dp[w][v] = dp[w - group[i]][max(0,v - profit[i])] + dp[w][v]

        Num = 0

        for w in range(G + 1):

            Num += dp[w][P]

        return Num % (10 ** 9 + 7)

    #  def profitableSchemes(self, G: int, P: int, group: List[int], profit: List[int]) -> int:
    def solve_3D(self, G: int, P: int, group, profit):
        """
        01 背包问题

        3维动态规划

        G=10^2
        n=10^2
        P=10^2

        时间复杂度 O(GnN)=10^6

        :return: 

        """

        n = len(group)

        group = [0] + group
        profit = [0] + profit

        # dp=np.zeros((n+1,G+1,P+1),dtype=int)

        dp = [[[0 for __ in range(P + 1)] for __ in range(G + 1)] for __ in range(n + 1)]

        dp[0][0][0] = 1

        for i in range(1, n + 1):
            for w in range(G + 1):
                for v in range(P + 1):

                    case1 = 0
                    if w - group[i] >= 0: #and v - profit[i] >= 0:

                        case1 = dp[i - 1][w - group[i]][max(0,v - profit[i])] #

                    case2 = dp[i - 1][w][v]

                    dp[i][w][v] = case1 + case2

        Num = 0

        for w in range(G + 1):

            Num += dp[n][w][P]

        return Num % (10 ** 9 + 7)

class Solution_Deprecated:

    #  def profitableSchemes(self, G: int, P: int, group: List[int], profit: List[int]) -> int:
    def solve_2D(self, G: int, P: int, group, profit):
        """
        01 背包问题

        2维动态规划
        状态压缩 

        G=10^2
        n=10^2

        时间复杂度 O(Gn)=10^4

        Wrong Answer !
        
        :param s: 
        :return: 

        """

        n = len(group)

        group = [0] + group
        profit = [0] + profit

        dpV = [0 for __ in range(G + 1)] # 达到 某重量时 背包中物品的总价值
        dpNum = [0 for __ in range(G + 1)] # 达到 某重量时 的 物品的组合 数

        dpV[0] = 0
        dpNum[0]=1

        Num=0

        for i in range(1, n + 1):
            for w in range(G, group[i] - 1, -1):

                if dpV[w - group[i]] + profit[i]>=P:
                    Num += dpNum[w - group[i]]

                # if  dpV[w]>=P:
                #     Num += dpNum[w]

                dpV[w] = max(dpV[w - group[i]] + profit[i],dpV[w])
                dpNum[w]+=dpNum[w - group[i]]


        return Num % (10 ** 9 + 7)

    #  def profitableSchemes(self, G: int, P: int, group: List[int], profit: List[int]) -> int:
    def solve_3D_opt(self, G: int, P: int, group, profit):
        """
        01 背包问题
        
        3维动态规划 (优化)
        状态压缩 
        
        G=10^2
        n=10^2
        N=10^4

        时间复杂度 O(GnN)=10^8
        
        还是 TLE 
        
        :param s: 
        :return: 

        """

        n = len(group)

        group = [0] + group
        profit = [0] + profit

        N = sum(profit[1:])

        # dp = np.zeros((G + 1, N + 1), dtype=int)

        dp= [[0 for __ in range(N+1)]for __ in range(G+1)]

        dp[0][0] = 1

        for i in range(1, n + 1):
            for w in range(G ,group[i]-1,-1):
                for v in range(N ,profit[i]-1,-1): # TODO: 降低运行时间 # time:  12.591132851000001 s -> time:  5.194653615 s

                    # print('w:{},v:{}'.format(w,v))

                    dp[w][v] = (dp[w - group[i]][v - profit[i]] + dp[w][v]) # % (10 ** 9 + 7)

        res = 0
        for w in range(G + 1):
            for v in range(P,N + 1):
                    res += dp[w][v]

        return res% (10 ** 9 + 7)

    #  def profitableSchemes(self, G: int, P: int, group: List[int], profit: List[int]) -> int:
    def solve_3D(self, G: int, P: int, group, profit):
        """
        01 背包问题
        
        3维动态规划
        
        G=10^2
        n=10^2
        N=10^4
        
        时间复杂度 O(GnN)=10^8
        
        TLE
                
        :param s: 
        :return: 
        
        """

        n=len(group)

        group=[0]+group
        profit = [0] + profit

        N=sum(profit[1:])

        # dp=np.zeros((n+1,G+1,N+1),dtype=int)

        dp= [[[0 for __ in range(N+1)]for __ in range(G+1)]for __ in range(n+1)]

        dp[0][0][0]=1

        for i in range(1,n+1):
            for w in range(G+1):
                for v in range(N+1):

                    case1=0
                    if w-group[i]>=0 and v-profit[i]>=0:
                        case1=dp[i-1][w-group[i]][v-profit[i]]

                    case2=dp[i-1][w][v]

                    dp[i][w][v]=case1+case2

        res=0
        for w in range(G + 1):
            for v in range(N + 1):
                if v>=P:
                    res+=dp[n][w][v]

        return res % (10**9+7)



class Test:

    def test_small_dataset(self, func):

        G = 10
        P = 5
        group = [2, 3, 5]
        profit = [6, 7, 8]
        assert func(G,P,group,profit) == 7

        G = 10
        P = 5
        group = [2, 3, 5]
        profit = [4, 2, 8]
        assert func(G,P,group,profit) == 5

        G = 5
        P = 3
        group = [2, 2]
        profit = [2, 3]
        assert func(G,P,group,profit) == 2

        G = 1
        P = 1
        group = [1]
        profit = [1]
        assert func(G,P,group,profit) == 1



        G =100
        P =100
        group =[10, 4, 3, 1, 6, 10, 7, 7, 4, 11, 20, 5, 13, 1, 27, 21, 1, 3, 1, 1, 1, 30, 7, 6, 5, 27, 39, 6, 18, 25, 4, 14, 2,
         3, 6, 2, 11, 3, 3, 8, 3, 13, 10, 3, 20, 34, 5, 48, 1, 3, 8, 41, 2, 1, 1, 1, 1, 19, 59, 2, 20, 18, 15, 1, 5, 10,
         16, 28, 4, 6, 1, 2, 1, 8, 15, 6, 5, 3, 18, 11, 11, 7, 34, 10, 1, 26, 1, 13, 7, 1, 9, 6, 37, 1, 32, 1, 9, 1, 1,
         8]
        profit =[9, 10, 0, 17, 22, 9, 1, 16, 12, 0, 8, 1, 12, 2, 5, 1, 0, 2, 12, 0, 18, 0, 11, 0, 3, 14, 0, 9, 0, 6, 0, 12, 1,
         17, 19, 7, 56, 28, 1, 4, 4, 3, 24, 4, 6, 1, 6, 10, 1, 1, 0, 2, 1, 13, 1, 6, 6, 9, 43, 6, 1, 0, 5, 10, 2, 7, 31,
         8, 3, 2, 3, 34, 7, 13, 4, 2, 1, 9, 13, 7, 9, 7, 15, 2, 10, 8, 3, 5, 3, 21, 22, 16, 1, 0, 0, 0, 1, 1, 1, 18]
        assert func(G, P, group, profit) == 900985844

        # TODO: 边界条件


        G = 1
        P = 1
        group = [1]
        profit = [1]
        assert func(G, P, group, profit) == 1

    def read_test_case_fromFile_matrix(self, dir):
        """
        解析矩阵

        :param dir: 
        :return: 
        """

        with open(dir, 'r', encoding='utf-8') as file:  #


            line_list = file.readline().strip()[2:-2].split('],[')

            matrix = []

            for line in line_list:
                matrix.append([int(ele) for ele in line.split(',')])

            print('matrix:', matrix)

            K = int(file.readline().strip())
            print('K: ', K)

            return K, matrix

    def read_test_case_fromFile_list(self,dir):
        """
        解析 列表
        
        :param dir: 
        :return: 
        """


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

        G=100
        P=100

        group=[18, 58, 88, 52, 54, 13, 50, 66, 83, 61, 100, 54, 60, 80, 1, 19, 78, 54, 67, 20, 57, 46, 12, 6, 14, 43, 64, 81,
         30, 60, 48, 53, 86, 71, 51, 23, 71, 87, 95, 69, 11, 12, 41, 36, 69, 89, 91, 10, 98, 31, 67, 85, 16, 83, 83, 14,
         14, 71, 33, 5, 40, 61, 22, 19, 34, 70, 50, 21, 91, 77, 4, 36, 16, 38, 56, 23, 68, 51, 71, 38, 63, 52, 14, 47,
         25, 57, 95, 35, 58, 32, 1, 39, 48, 33, 89, 9, 1, 95, 90, 78]

        profit=[96, 77, 37, 98, 66, 44, 18, 37, 47, 9, 38, 82, 74, 12, 71, 31, 80, 64, 15, 45, 85, 52, 70, 53, 94, 90, 90, 14,
         98, 22, 33, 39, 18, 22, 10, 46, 6, 19, 25, 50, 33, 15, 63, 93, 35, 0, 76, 44, 37, 68, 35, 80, 70, 66, 4, 88,
         66, 93, 49, 19, 25, 90, 21, 59, 17, 40, 46, 79, 5, 41, 2, 37, 27, 92, 0, 53, 57, 91, 75, 0, 42, 100, 16, 97,
         83, 75, 57, 61, 73, 21, 63, 97, 75, 95, 84, 14, 98, 47, 0, 13]


        start = timeit.default_timer()
        print('run large dataset: ')
        func(G,P,group,profit)
        end = timeit.default_timer()
        print('time: ', end - start, 's')
        print('copy the test case to leetcode to judge the time complex')

        G =100
        P =100
        group =[10, 4, 3, 1, 6, 10, 7, 7, 4, 11, 20, 5, 13, 1, 27, 21, 1, 3, 1, 1, 1, 30, 7, 6, 5, 27, 39, 6, 18, 25, 4, 14, 2,
         3, 6, 2, 11, 3, 3, 8, 3, 13, 10, 3, 20, 34, 5, 48, 1, 3, 8, 41, 2, 1, 1, 1, 1, 19, 59, 2, 20, 18, 15, 1, 5, 10,
         16, 28, 4, 6, 1, 2, 1, 8, 15, 6, 5, 3, 18, 11, 11, 7, 34, 10, 1, 26, 1, 13, 7, 1, 9, 6, 37, 1, 32, 1, 9, 1, 1,
         8]
        profit =[9, 10, 0, 17, 22, 9, 1, 16, 12, 0, 8, 1, 12, 2, 5, 1, 0, 2, 12, 0, 18, 0, 11, 0, 3, 14, 0, 9, 0, 6, 0, 12, 1,
         17, 19, 7, 56, 28, 1, 4, 4, 3, 24, 4, 6, 1, 6, 10, 1, 1, 0, 2, 1, 13, 1, 6, 6, 9, 43, 6, 1, 0, 5, 10, 2, 7, 31,
         8, 3, 2, 3, 34, 7, 13, 4, 2, 1, 9, 13, 7, 9, 7, 15, 2, 10, 8, 3, 5, 3, 21, 22, 16, 1, 0, 0, 0, 1, 1, 1, 18]

        start = timeit.default_timer()
        print('run large dataset: ')
        func(G,P,group,profit) # 900985844
        end = timeit.default_timer()
        print('time: ', end - start, 's')
        print('copy the test case to leetcode to judge the time complex')

        # dir = 'large_test_case/188_1'
        # K, l1 = self.read_test_case_fromFile_list(dir)
        #
        # start = timeit.default_timer()
        # print('run large dataset:{} '.format(dir))
        # func(K, l1)  # 12.047259273 s
        # end = timeit.default_timer()
        # print('time: ', end - start, 's')


if __name__ == '__main__':

    sol = Solution_Deprecated()

    # IDE 测试 阶段：

    G = 10
    P = 5
    group = [2, 3, 5]
    profit = [6, 7, 8]

    # print(sol.solve_3D(G,P,group,profit))


    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve_2D)

    # test.test_large_dataset(sol.solve_3D) # time:  17.865607367000003 s

    # test.test_large_dataset(sol.solve_3D_opt) # time:  5.296859553 s

    # test.test_large_dataset(sol.solve_2D) #

    sol2 = Solution()

    # IDE 测试 阶段：

    G = 10
    P = 5
    group = [2, 3, 5]
    profit = [6, 7, 8]

    print(sol2.solve_3D_opt(G,P,group,profit))

    test.test_small_dataset(sol2.solve_3D_opt)
    #
    test.test_large_dataset(sol2.solve_3D_opt)







