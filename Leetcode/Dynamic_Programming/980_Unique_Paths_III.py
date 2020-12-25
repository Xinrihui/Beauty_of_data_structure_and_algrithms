#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class Solution:

    # def uniquePathsIII(self, grid: List[List[int]]) -> int:

    def solve(self, grid):
        """
        回溯法

        时间复杂度 O(n!) 


        :param grid: 
        :return: 
        """

        start=None
        self.end=None

        self.obs=set()
        unobs=set()

        self.m=len(grid)
        self.n=len(grid[0])

        for i in range(self.m):
            for j in range(self.n):

                if grid[i][j]==1:
                    start=(i,j)
                    unobs.add((i, j))

                elif  grid[i][j]==2:
                    self.end=(i,j)
                    unobs.add((i, j))

                elif grid[i][j]==0:
                    unobs.add((i,j))

                elif grid[i][j]==-1:
                    self.obs.add((i,j))

        self.L=len(unobs)

        self.Num=0

        curr = start
        l = 1
        visit = set()

        path=[curr]

        self.__process(curr, l, visit,path)

        return self.Num

    def __process(self,curr, l, visit,path):

        # print(curr,path)

        if curr in visit | self.obs : # 访问过 遇到障碍

            return


        if curr == self.end:
            print('reach the end: {}'.format(path))

            if l== self.L:
                print('length is L ')
                self.Num+=1

            return

        x=curr[0]
        y=curr[1]

        visit=visit|set([(x,y)])

        if x+1 < self.m : # down

            next=(x+1,y)
            self.__process(next,l+1,visit,path+[next])

        if x-1 >=0  : # up

            next = (x-1,y)
            self.__process(next, l + 1, visit, path + [next])

        if y-1 >=0  : # left

            next = (x,y-1)
            self.__process(next, l + 1, visit, path + [next])

        if y+1 <self.n  : # right

            next=(x,y+1)
            self.__process(next, l + 1, visit, path + [next])





class Test:
    def test_small_dataset(self, func):


        grid = [[1,0,0,0],[0,0,0,0],[0,0,2,-1]]
        assert func(grid) == 2

        grid=[[1,0,0,0],[0,0,0,0],[0,0,0,2]]
        assert func(grid) == 4

        grid=[[0,1],[2,0]]
        assert func(grid) == 0

        # TODO: 边界条件
        # assert func(None) == None

        # assert func(1, 0) == True

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

    def read_test_case_fromFile_list(self, dir):
        """
        解析 列表

        :param dir: 
        :return: 
        """

        with open(dir, 'r', encoding='utf-8') as file:  #

            K1 = int(file.readline().strip())
            print('K1: ', K1)

            l1 = file.readline().strip()[1:-1].split(',')

            l1 = [int(ele) for ele in l1]

            print('l1:', l1)

            return K1, l1

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

    grid=[[1,0,0,0],
          [0,0,0,0],
          [0,0,2,-1]]

    # grid=[[1,0],
    #       [0,2]]

    # print(sol.solve(grid))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










