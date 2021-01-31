#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

class Solution2:

    #  def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
    def solve(self, price,special,needs):
        """
        price=[2,5], special=[[3,0,5],[1,2,10]], needs=[3,2]
        
        记忆化递归 
        
        时间复杂度  O( K*(N^m) )

        :return: 
        """

        self.dp={}

        N=len(price) # 物品的个数

        needs=tuple(needs)

        return self.__process(N,price,special,needs)

    def __process(self,N, price,special,needs):

        if needs in self.dp:
            return self.dp[needs]

        # 原价购买
        orgin_price= sum([ price[i]*needs[i] for i in range(N) ])

        # 买大礼包
        min_price=float('inf')

        for offer in special:

            value=offer[-1]
            items=offer[0: N]

            diff = tuple([ needs[i]-items[i] for i in range(N) ])

            if min(diff)>=0: # 礼包 中的 各个物品的个数 不能超过 need 中的个数

                min_price=min( self.__process(N,price,special,diff)+value ,min_price)

        min_price=min(min_price,orgin_price)

        self.dp[needs]=min_price

        return min_price


class Solution:

    #  def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
    def solve(self, price, special, needs):
        """
        price=[2,5], special=[[3,0,5],[1,2,10]], needs=[3,2]

        动态规划

        时间复杂度  O( K*(N^m) )
        
        TLE 
        
        :return: 
        """

        self.dp = {}

        N = len(price)  # 物品的个数

        needs = tuple(needs)

        # 生成 所有 礼包的列表( 把 单个物品 也看成一种礼包)
        items=[]
        values=[]

        for i in range(N): #
            ele= [0]*N
            ele[i]=1
            items.append(tuple(ele))
            values.append(price[i])

        for offer in special:
            items.append(tuple(offer[0:N]))
            values.append(offer[-1])


        # print(items) # [(1, 0), (0, 1), (3, 0), (1, 2)]
        # print(values) # [2, 5, 5, 10]

        # N 重 背包问题
        # 生成 所有 背包的 状态
        w = [range(i + 1) for i in needs]
        states=self.product(w)

        # print(states)
        # [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1], [0, 2], [1, 2], [2, 2], [3, 2]]

        # 初始化
        dp={tuple(state):float('inf') for state in states}

        start=tuple([0]*N)
        dp[start]=0

        for state in states:

            state=tuple(state)

            if  state == start:
                continue

            v_min=float('inf')

            for j in range(len(items)):

                bag=items[j]

                diff= tuple([ state[i]-bag[i] for i in range(N)]) # TODO: TLE

                if min(diff)>=0:
                    v_min=min(v_min, dp[diff]+values[j])

            dp[state]=v_min

        # print(dp)

        return dp[tuple(needs)]


    def product(self, lists ):

        """
        笛卡尔积(向量叉乘)
        eg.
         product(['ABCD', 'xy']) --> Ax Ay Bx By Cx Cy Dx Dy
        
        :param lists: 
        :return: 
        """

        pools = [tuple(pool) for pool in lists]   # [('A', 'B', 'C', 'D'), ('x', 'y')]

        # print('pools:', pools)

        result = [[]]

        # for pool in pools:
        #     result = [x + [y] for x in result for y in pool]

        for pool in pools:
            # print('pool:',pool)
            res = []
            for x in pool:
                # print(x)
                for y in result:
                    # print(y)
                    res.append( y+[x])
            # print('res:',res)  # [['A'], ['B'], ['C'], ['D']]
                        #  [['A', 'x'], ['B', 'x'], ['C', 'x'], ['D', 'x'], ['A', 'y'], ['B', 'y'], ['C', 'y'], ['D', 'y']]
            result = res

        # print('result:', result)

        return result



class Test:
    def test_small_dataset(self, func):

        price = [2, 5]
        special = [[3, 0, 5], [1, 2, 10]]
        needs = [3, 2]

        assert func(price, special, needs) == 14


        assert func([2, 3, 4], [[1, 1, 0, 4], [2, 2, 1, 9]], [1, 2, 1]) == 11

        price=[9, 6, 1, 5, 3, 4]
        special=[[1, 2, 2, 1, 0, 4, 14], [6, 3, 4, 0, 0, 1, 16], [4, 5, 6, 6, 2, 4, 26], [1, 1, 4, 3, 4, 3, 15],
         [4, 2, 5, 4, 4, 5, 15], [4, 0, 0, 2, 3, 5, 13], [2, 4, 6, 4, 3, 5, 7], [3, 3, 4, 2, 2, 6, 21],
         [0, 3, 0, 2, 3, 3, 15], [0, 2, 4, 2, 2, 5, 24], [4, 1, 5, 4, 5, 4, 25], [6, 0, 5, 0, 1, 1, 14],
         [4, 0, 5, 2, 1, 5, 8], [4, 1, 4, 4, 3, 1, 10], [4, 4, 2, 1, 5, 0, 14], [2, 4, 4, 1, 3, 1, 16],
         [4, 2, 3, 1, 2, 1, 26], [2, 4, 1, 6, 5, 3, 2], [0, 2, 0, 4, 0, 0, 19], [3, 1, 6, 3, 3, 1, 23],
         [6, 2, 3, 2, 4, 4, 16], [5, 3, 5, 5, 0, 4, 5], [5, 0, 4, 3, 0, 2, 20], [5, 3, 1, 2, 2, 5, 8],
         [3, 0, 6, 1, 0, 2, 10], [5, 6, 6, 1, 0, 4, 12], [0, 6, 6, 4, 6, 4, 21], [0, 4, 6, 5, 0, 0, 22],
         [0, 4, 2, 4, 4, 6, 16], [4, 2, 1, 0, 6, 5, 14], [0, 1, 3, 5, 0, 3, 8], [5, 5, 3, 3, 2, 0, 4],
         [1, 0, 3, 6, 2, 3, 18], [4, 2, 6, 2, 2, 5, 2], [0, 2, 5, 5, 3, 6, 12], [1, 0, 6, 6, 5, 0, 10],
         [6, 0, 0, 5, 5, 1, 24], [1, 4, 6, 5, 6, 3, 19], [2, 2, 4, 2, 4, 2, 20], [5, 6, 1, 4, 0, 5, 3],
         [3, 3, 2, 2, 1, 0, 14], [0, 1, 3, 6, 5, 0, 9], [5, 3, 6, 5, 3, 3, 11], [5, 3, 3, 1, 0, 2, 26],
         [0, 1, 1, 4, 2, 1, 16], [4, 2, 3, 2, 1, 4, 6], [0, 2, 1, 3, 3, 5, 15], [5, 6, 4, 1, 2, 5, 18],
         [1, 0, 0, 1, 6, 1, 16], [2, 0, 6, 6, 2, 2, 17], [4, 4, 0, 2, 4, 6, 12], [0, 5, 2, 5, 4, 6, 6],
         [5, 2, 1, 6, 2, 1, 24], [2, 0, 2, 2, 0, 1, 14], [1, 1, 0, 5, 3, 5, 16], [0, 2, 3, 5, 5, 5, 6],
         [3, 2, 0, 6, 4, 6, 8], [4, 0, 1, 4, 5, 1, 6], [5, 0, 5, 6, 6, 3, 7], [2, 6, 0, 0, 2, 1, 25],
         [0, 4, 6, 1, 4, 4, 6], [6, 3, 1, 4, 1, 1, 24], [6, 2, 1, 2, 1, 4, 4], [0, 1, 2, 3, 0, 1, 3],
         [0, 2, 5, 6, 5, 2, 13], [2, 6, 4, 2, 2, 3, 17], [3, 4, 5, 0, 5, 4, 20], [6, 2, 3, 4, 1, 3, 4],
         [6, 4, 0, 0, 0, 5, 16], [3, 1, 2, 5, 0, 6, 11], [1, 3, 2, 2, 5, 6, 14], [1, 3, 4, 5, 3, 5, 18],
         [2, 1, 1, 2, 6, 1, 1], [4, 0, 4, 0, 6, 6, 8], [4, 6, 0, 5, 0, 2, 1], [3, 1, 0, 5, 3, 2, 26],
         [4, 0, 4, 0, 6, 6, 6], [5, 0, 0, 0, 0, 4, 26], [4, 3, 2, 2, 0, 2, 14], [5, 2, 4, 0, 2, 2, 26],
         [3, 4, 6, 0, 2, 4, 25], [2, 1, 5, 5, 1, 3, 26], [0, 5, 2, 4, 0, 2, 24], [5, 2, 5, 4, 5, 0, 1],
         [5, 3, 0, 1, 5, 4, 15], [6, 1, 5, 1, 2, 1, 21], [2, 5, 1, 2, 1, 4, 15], [1, 4, 4, 0, 0, 0, 1],
         [5, 0, 6, 1, 1, 4, 22], [0, 1, 1, 6, 1, 4, 1], [1, 6, 0, 3, 2, 2, 17], [3, 4, 3, 3, 1, 5, 17],
         [1, 5, 5, 4, 5, 2, 27], [0, 6, 5, 5, 0, 0, 26], [1, 4, 0, 3, 1, 0, 13], [1, 0, 3, 5, 2, 4, 5],
         [2, 2, 2, 3, 0, 0, 11], [3, 2, 2, 1, 1, 1, 6], [6, 6, 1, 1, 1, 6, 26], [1, 5, 1, 2, 5, 2, 12]]
        needs=[6, 6, 6, 1, 6, 6]

        assert func(price, special, needs) ==34

        # TODO: 边界条件
        # assert func(None) == None




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

        price=[9, 6, 1, 5, 3, 4]
        special=[[1, 2, 2, 1, 0, 4, 14], [6, 3, 4, 0, 0, 1, 16], [4, 5, 6, 6, 2, 4, 26], [1, 1, 4, 3, 4, 3, 15],
         [4, 2, 5, 4, 4, 5, 15], [4, 0, 0, 2, 3, 5, 13], [2, 4, 6, 4, 3, 5, 7], [3, 3, 4, 2, 2, 6, 21],
         [0, 3, 0, 2, 3, 3, 15], [0, 2, 4, 2, 2, 5, 24], [4, 1, 5, 4, 5, 4, 25], [6, 0, 5, 0, 1, 1, 14],
         [4, 0, 5, 2, 1, 5, 8], [4, 1, 4, 4, 3, 1, 10], [4, 4, 2, 1, 5, 0, 14], [2, 4, 4, 1, 3, 1, 16],
         [4, 2, 3, 1, 2, 1, 26], [2, 4, 1, 6, 5, 3, 2], [0, 2, 0, 4, 0, 0, 19], [3, 1, 6, 3, 3, 1, 23],
         [6, 2, 3, 2, 4, 4, 16], [5, 3, 5, 5, 0, 4, 5], [5, 0, 4, 3, 0, 2, 20], [5, 3, 1, 2, 2, 5, 8],
         [3, 0, 6, 1, 0, 2, 10], [5, 6, 6, 1, 0, 4, 12], [0, 6, 6, 4, 6, 4, 21], [0, 4, 6, 5, 0, 0, 22],
         [0, 4, 2, 4, 4, 6, 16], [4, 2, 1, 0, 6, 5, 14], [0, 1, 3, 5, 0, 3, 8], [5, 5, 3, 3, 2, 0, 4],
         [1, 0, 3, 6, 2, 3, 18], [4, 2, 6, 2, 2, 5, 2], [0, 2, 5, 5, 3, 6, 12], [1, 0, 6, 6, 5, 0, 10],
         [6, 0, 0, 5, 5, 1, 24], [1, 4, 6, 5, 6, 3, 19], [2, 2, 4, 2, 4, 2, 20], [5, 6, 1, 4, 0, 5, 3],
         [3, 3, 2, 2, 1, 0, 14], [0, 1, 3, 6, 5, 0, 9], [5, 3, 6, 5, 3, 3, 11], [5, 3, 3, 1, 0, 2, 26],
         [0, 1, 1, 4, 2, 1, 16], [4, 2, 3, 2, 1, 4, 6], [0, 2, 1, 3, 3, 5, 15], [5, 6, 4, 1, 2, 5, 18],
         [1, 0, 0, 1, 6, 1, 16], [2, 0, 6, 6, 2, 2, 17], [4, 4, 0, 2, 4, 6, 12], [0, 5, 2, 5, 4, 6, 6],
         [5, 2, 1, 6, 2, 1, 24], [2, 0, 2, 2, 0, 1, 14], [1, 1, 0, 5, 3, 5, 16], [0, 2, 3, 5, 5, 5, 6],
         [3, 2, 0, 6, 4, 6, 8], [4, 0, 1, 4, 5, 1, 6], [5, 0, 5, 6, 6, 3, 7], [2, 6, 0, 0, 2, 1, 25],
         [0, 4, 6, 1, 4, 4, 6], [6, 3, 1, 4, 1, 1, 24], [6, 2, 1, 2, 1, 4, 4], [0, 1, 2, 3, 0, 1, 3],
         [0, 2, 5, 6, 5, 2, 13], [2, 6, 4, 2, 2, 3, 17], [3, 4, 5, 0, 5, 4, 20], [6, 2, 3, 4, 1, 3, 4],
         [6, 4, 0, 0, 0, 5, 16], [3, 1, 2, 5, 0, 6, 11], [1, 3, 2, 2, 5, 6, 14], [1, 3, 4, 5, 3, 5, 18],
         [2, 1, 1, 2, 6, 1, 1], [4, 0, 4, 0, 6, 6, 8], [4, 6, 0, 5, 0, 2, 1], [3, 1, 0, 5, 3, 2, 26],
         [4, 0, 4, 0, 6, 6, 6], [5, 0, 0, 0, 0, 4, 26], [4, 3, 2, 2, 0, 2, 14], [5, 2, 4, 0, 2, 2, 26],
         [3, 4, 6, 0, 2, 4, 25], [2, 1, 5, 5, 1, 3, 26], [0, 5, 2, 4, 0, 2, 24], [5, 2, 5, 4, 5, 0, 1],
         [5, 3, 0, 1, 5, 4, 15], [6, 1, 5, 1, 2, 1, 21], [2, 5, 1, 2, 1, 4, 15], [1, 4, 4, 0, 0, 0, 1],
         [5, 0, 6, 1, 1, 4, 22], [0, 1, 1, 6, 1, 4, 1], [1, 6, 0, 3, 2, 2, 17], [3, 4, 3, 3, 1, 5, 17],
         [1, 5, 5, 4, 5, 2, 27], [0, 6, 5, 5, 0, 0, 26], [1, 4, 0, 3, 1, 0, 13], [1, 0, 3, 5, 2, 4, 5],
         [2, 2, 2, 3, 0, 0, 11], [3, 2, 2, 1, 1, 1, 6], [6, 6, 1, 1, 1, 6, 26], [1, 5, 1, 2, 5, 2, 12]]
        needs=[6, 6, 6, 1, 6, 6]

        start = timeit.default_timer()
        print('run large dataset: ')
        func(price, special, needs)
        end = timeit.default_timer()
        print('time: ', end - start, 's')




if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：
    price = [2, 5]
    special = [[3, 0, 5], [1, 2, 10]]
    needs = [3, 2]

    # print(sol.solve(price,special,needs))

    sol2=Solution2()

    # print(sol2.product(['ABCD', 'xy']))

    # print(sol2.product([range(4),range(3) ]))

    # print(sol2.solve(price,special,needs))

    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    test.test_large_dataset(sol.solve)

    test.test_large_dataset(sol2.solve)










