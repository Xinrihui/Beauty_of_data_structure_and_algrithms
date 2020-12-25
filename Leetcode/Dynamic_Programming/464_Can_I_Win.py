#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

class Solution_depreacated:

    #   def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
    def solve(self, maxChoosableInteger,desiredTotal):
        """
        回溯法

        时间复杂度 O(n!) n=maxChoosableInteger
        
        TLE 

        :param s: 
        :return: 
        """

        self.T=desiredTotal
        self.S=set(range(1,maxChoosableInteger+1))

        if sum(self.S)<self.T:
            return False

        win=False
        t=0
        visit=set()

        for c in self.S:

            win= win or self.__process(c,t+c,visit|set([c]))

        return win


    def __process(self,c,t,visit):


        if t >= self.T:

            # print('win! c:{},t:{},visit:{}'.format(c, t, visit))

            return True

        # print('c:{},t:{},visit:{}'.format(c,t,visit))

        nextWin=False
        for current in self.S-visit:

            # print('current:{}'.format(current))

            nextWin= nextWin or self.__process(current,t+current,visit|set([current]))
                    # 若 nextWin 为 ture 则 后面的 表达式 不再计算

            # print('nextWin:',nextWin)

        win=not nextWin

        return win




class Solution:
    #   def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
    def solve(self, N, T):
        """
        带缓存的 回溯法 (记忆化递归)

        1.缓存 子问题的解
        2. 缓存 中的 状态 采用 基于位 的压缩表示 

        时间复杂度 O(2^N) 

        :return: 
        """

        self.T = T
        self.S = set(range(1, N + 1))

        if T == 0:  # 目标值为0 显然A 直接获胜
            return True

        if sum(self.S) < self.T:  # 目标值 比 所有元素 求和 还大，则 判定 A 输了
            return False

        self.dp = [None] * (2 ** N)

        c = None
        t = 0  # 累计和
        visit = set()

        state = 0

        win = not self.__process(c, visit, t, state)

        return win

    def __process(self, c, visit, t, state):

        if self.dp[state] != None:
            return self.dp[state]

        if t >= self.T:  # 递归结束 条件
            # print('win! c:{},t:{},visit:{}'.format(c, t, visit))
            return True

        # print('c:{},t:{},visit:{}'.format(c,t,visit))

        nextWin = False

        visit = visit | set([c])

        for next in self.S - visit:
            # print('next:{}'.format(next))

            nextWin = nextWin or self.__process(next, visit, t + next, state | (1 << (next - 1)))
            # 若 nextWin 为 ture 则 后面的 表达式 不再计算

            # print('nextWin:',nextWin)

        win = not nextWin

        self.dp[state] = win

        return win


class Test:
    def test_small_dataset(self, func):

        assert func(3,4) == False

        assert func(10,11) == False

        assert func(5,7) == True

        assert func(6, 4) == True

        assert func(18, 79) == True

        assert func(5, 50) == False

        assert func(19, 190) == True


        # TODO: 边界条件
        # assert func(None) == None

        assert func(1,0) == True


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

        N = int(2 * pow(10, 4))
        max_v = int(pow(10, 9))

        l = np.random.randint(max_v, size=N)
        l1 = list(l)

        start = timeit.default_timer()
        print('run large dataset: ')
        func()
        end = timeit.default_timer()
        print('time: ', end - start, 's')


        dir = 'large_test_case/188_1'
        K, l1 = self.read_test_case_fromFile_list(dir)

        start = timeit.default_timer()
        print('run large dataset:{} '.format(dir))
        func(K, l1)  # 12.047259273 s
        end = timeit.default_timer()
        print('time: ', end - start, 's')
        print('copy the test case to leetcode to judge the time complex')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    # print(sol.solve(3,4))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










