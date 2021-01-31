#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

class Solution_deprecated:

    #  def maxStudents(self, seats: List[List[str]]) -> int:
    def solve(self, seats):
        """
        
        记忆化递归
        
        自顶向下
        
        时间复杂度 O( 2^(m*n) )
        
        1 <= m <= 8
        1 <= n <= 8
        
        TLE

        :param seats: 
        :return: 
        """

        m=len(seats)
        n=len(seats[0])

        pos_list=[]

        idx=0
        for i in range(m):
            for j in range(n):
                if seats[i][j]=='.':

                    seats[i][j]=idx
                    pos_list.append((idx,i,j))
                    idx += 1

        print(seats)
        print(pos_list)

        Num=len(pos_list) # 6

        No_pos={i:0 for i in range(Num)}

        for idx,i,j in pos_list:

            if j-1>=0: # 左边 是否 有 座位

                if seats[i][j-1] !='#':

                    idx2=seats[i][j-1]
                    No_pos[idx]=  No_pos[idx] | (1 << idx2)
                    No_pos[idx2] = No_pos[idx2] | (1 << idx)

            if j+1<n: # 右边是否 有座位

                if seats[i][j + 1] != '#':
                    idx2 = seats[i][j + 1]
                    No_pos[idx] = No_pos[idx] | (1 << idx2)
                    No_pos[idx2] = No_pos[idx2] | (1 << idx)

            if i-1>=0 and j-1>=0:  # 左上 是否 有座位

                if seats[i-1][j-1] != '#':
                    idx2 = seats[i-1][j-1]
                    No_pos[idx] = No_pos[idx] | (1 << idx2)
                    No_pos[idx2] = No_pos[idx2] | (1 << idx)

            if i - 1 >= 0 and j + 1 <n:  # 右上 是否 有座位

                if seats[i - 1][j + 1] != '#':
                    idx2 = seats[i - 1][j + 1]
                    No_pos[idx] = No_pos[idx] | (1 << idx2)
                    No_pos[idx2] = No_pos[idx2] | (1 << idx)


        print(No_pos)

        self.dp={}

        state= 2**Num-1 # 2^6-1 = 63

        l=Num
        self.__process(Num,No_pos,state)

        # print('dp:',self.dp)

        res= max(self.dp.values())

        return res

    def __process(self,Num,No_pos,state):

        if state in self.dp:
            return  self.dp[state]

        max_l=float('-inf')

        if state==0:

            max_l=0

        else:

            for i in range(Num):

                if (state>>i) & 1 ==1: # 若 第 i 位 为 1

                    next=i
                    nextState= state ^ (1<<i) # 将第 i 位 置为 0

                    if nextState & No_pos[next]==0:
                        c= self.__process(Num,No_pos,nextState)+1

                    else:
                        c=float('-inf')
                        self.__process(Num, No_pos, nextState) # 继续 往下搜索

                    max_l = max(c, max_l)


        self.dp[state]=max_l

        return max_l


class Solution:

    #  def maxStudents(self, seats: List[List[str]]) -> int:
    def solve(self, seats):
        """

        动态规划
        
        状态压缩 

        时间复杂度 O( m*(2^n) )


        :param seats: 
        :return: 
        """

        m = len(seats)
        n = len(seats[0])

        N=2**n

        dp=np.zeros((m,N),dtype=int)

        # dp=[[0 for __ in range(N)] for __ in range(m)]


        # 预处理：
        # 只考虑 座位的好坏 每一行所有 合法状态的集合
        states=[[] for __ in range(m)]

        for i in range(m):
            for j in range(N):  # 第 i行的 状态

                flag=True
                for p in range(n):  # 遍历 第 i行的状态 的每一位

                    if (j >> p) & 1 == 1 and seats[i][p]=='#':  # 若 第 p 位 为 1 但是 凳子却是坏的,则此状态非法
                        flag=False
                        break

                if  flag==True:
                    states[i].append(j)

        # print(states)

        # [[0, 2, 16, 18],
        #  [0, 1, 32, 33],
        #  [0, 2, 16, 18]]

        # 初始化 i=0
        for j in states[0]:

            bit_num = self.countSetBits(j)
            left_right = (j & (j >> 1) == 0)  # j 中没有 连续的 bit=1

            if left_right:# j 状态合法
                dp[0][j]=bit_num


        for i in range(1,m):
            for j in states[i]: # 当前行的 状态

                bit_num=self.countSetBits(j)

                for k in states[i-1]: #  上一行的状态

                    left_right= ( j&(j>>1)==0 ) # j 中没有 连续的 bit=1

                    up_left_right= (k & (j>>1)==0 ) and ( j & (k>>1)==0 ) # 左上 和 右上 没有 bit=1

                    if left_right and up_left_right:
                        dp[i][j]=max( dp[i][j], dp[i-1][k]+bit_num )

        # print(dp)

        res=max(dp[m-1])

        return res

    def  countSetBits(self,n):
        """
         统计 n 中 bit 为 1 的个数
        :param n: 
        :return: 
        """
        count = 0
        while (n):
            count += n & 1
            n >>= 1
        return count

class Test:
    def test_small_dataset(self, func):

        seats = [["#", ".", "#", "#", ".", "#"],
                 [".", "#", "#", "#", "#", "."],
                 ["#", ".", "#", "#", ".", "#"]]

        assert func(seats) == 4

        seats = [[".", "#"],
                 ["#", "#"],
                 ["#", "."],
                 ["#", "#"],
                 [".", "#"]]

        assert func(seats) == 3

        seats = [["#", ".", ".", ".", "#"],
                 [".", "#", ".", "#", "."],
                 [".", ".", "#", ".", "."],
                 [".", "#", ".", "#", "."],
                 ["#", ".", ".", ".", "#"]]

        assert func(seats) == 10

        seats=[[".", "#", "#", ".", "#", "#", "#"],
               [".", "#", "#", ".", ".", ".", "."],
               ["#", "#", ".", ".", "#", "#", "#"],
                [".", ".", ".", "#", "#", ".", "."],
               [".", "#", "#", ".", ".", ".", "#"],
               [".", ".", ".", ".", ".", "#", "."]]

        assert func(seats) == 14

        seats=[[".", ".", "#", "#"], [".", "#", ".", "."], ["#", ".", ".", "#"], ["#", "#", "#", "."]]

        assert func(seats) == 4

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

        seats=[[".", "#", "#", ".", "#", "#", "#"],
               [".", "#", "#", ".", ".", ".", "."],
               ["#", "#", ".", ".", "#", "#", "#"],
                [".", ".", ".", "#", "#", ".", "."],
               [".", "#", "#", ".", ".", ".", "#"],
               [".", ".", ".", ".", ".", "#", "."]]

        start = timeit.default_timer()
        print('run large dataset: ')
        func(seats)
        end = timeit.default_timer()
        print('time: ', end - start, 's')


        # dir = 'large_test_case/188_1'
        # K, l1 = self.read_test_case_fromFile_list(dir)
        #
        # start = timeit.default_timer()
        # print('run large dataset:{} '.format(dir))
        # func(K, l1)  # 12.047259273 s
        # end = timeit.default_timer()
        # print('time: ', end - start, 's')
        # print('copy the test case to leetcode to judge the time complex')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：

    # seats = [["#", ".", "#", "#", ".", "#"],
    #          [".", "#", "#", "#", "#", "."],
    #          ["#", ".", "#", "#", ".", "#"]]

    seats = [[".", ".", "#", "#"],
             [".", "#", ".", "."],
             ["#", ".", ".", "#"],
             ["#", "#", "#", "."]]

    print(sol.solve(seats))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










