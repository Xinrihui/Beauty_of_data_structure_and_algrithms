#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

class Solution:

    #  def prisonAfterNDays(self, cells: List[int], N: int) -> List[int]:
    def solve(self, cells,N):
        """

        时间复杂度 

        :param cells: 
        :param N:
         
        :return: 
        """

        # print(self.next_day_cell(cells))

        state=cells
        dp={}

        dp[tuple(state)]=0 #

        first_state=None # 出现 第一个 重复的状态
        idx=None

        for i in range(1,N+1):

            next_state= tuple(self.next_day_cell(state))

            # print('i:{},state:{}'.format(i,next_state))

            if next_state in dp: # 找到 第一个重复的状态
                first_state=next_state
                idx=i

                break

            dp[next_state]=i
            state=next_state

        else: # N 太小了, 还没有出现 重复的状态
            return list(next_state)

        prev_idx=dp[first_state] # 重复状态 第一次出现的位置

        T=(idx-prev_idx) # 状态 出现的周期 T
        steps=(N-idx) % T  # 跳过 这些周期, 从 idx 出发 还需走 steps 步

        state=first_state

        for i in range(idx,idx+steps):
            next_state=self.next_day_cell(state)
            state=next_state

        return list(state)


    def next_day_cell(self,current):
        """
        
        :param current: 
        :return: 
        """

        N=len(current)

        next_state=[0]*N  # 首尾 元素 一定为0

        for i in range(1,N-1):# 1,2,...,N-2
            if (current[i-1]==1 and current[i+1]==1) or (current[i-1]==0 and current[i+1]==0):
                next_state[i]=1

        return next_state



class Test:
    def test_small_dataset(self, func):

        cells = [0, 1, 0, 1, 1, 0, 0, 1]
        N = 7
        assert func(cells,N) == [0,0,1,1,0,0,0,0]

        cells = [1, 0, 0, 1, 0, 0, 1, 0]
        N = 1000000000
        assert func(cells,N) == [0,0,1,1,1,1,1,0]


        # TODO: 边界条件
        # assert func(None) == None
        #
        # assert func('') == ''


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

    cells = [0, 1, 0, 1, 1, 0, 0, 1]
    N = 7

    print(sol.solve(cells,N))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










