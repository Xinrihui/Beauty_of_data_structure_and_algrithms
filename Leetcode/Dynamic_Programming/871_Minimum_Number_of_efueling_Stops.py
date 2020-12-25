#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

class Solution_deprecated1:

    # def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]])
    def solve(self, target, startFuel, stations):
        """
        TLE
        
        回溯法
        
        缓存子问题
        
        时间复杂度  O(n!)

        :param s: 
        :return: 
        """
        stations=[[0,startFuel]]+stations+[[target,0]] # 起点 和 终点 加入 stations

        # print(stations)
        # [[0, 10], [10, 60], [20, 30], [30, 30], [60, 40], [100, 0]]

        self.cache={}

        N=len(stations)

        idx=0
        pos=stations[idx][0] # 位置 0
        k=stations[idx][1] # 邮箱中的 剩余的油 数量 10

        times=self.__process(stations,N,idx,pos,k)-1 # 起点 加的油 要减去

        print(self.cache)

        if times == float('inf'):
            return -1

        return times

    def __process(self,stations,N,idx,pos,k):


        if (pos,k) not in self.cache:

            if idx == N - 1:  # 到达 最后一个 站 即 终点

                self.cache[(pos, k)] = 0

                return 0

            min_times=float('inf')

            for i in range(idx+1,N):

                c_pos = stations[i][0]  # 到达 stations[i] 的位置
                c_k=k-(stations[i][0]-pos) # 到达 stations[i] 油箱 剩余的油量

                if c_k>=0: # 到达 stations[i] 剩余的油量 必须 >=0

                    c_k_add = c_k + stations[i][1]  # 到达 stations[i] 并加油后的 油箱的 油量

                    times=self.__process(stations,N,i,c_pos,c_k_add)
                    min_times = min(min_times, times)

                else:
                    times=float('inf')

                    # self.cache[(c_pos, c_k)] = times

                    min_times = min(min_times, times)

                    break # 后面的 加油站更远 当前的油不够用 不用再考察了


            min_times=min_times+1

            self.cache[(pos,k)]=min_times

            return min_times

        else:
            print('({},{}) has cached'.format(pos,k))
            return self.cache[(pos,k)]


class Solution_deprecated2:

    # def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]])
    def solve(self, target, startFuel, stations):
        """
        Wrong Answer
        
        回溯法 
        缓存子问题 + 剪枝
        
        时间复杂度 

        :param s: 
        :return: 
        """
        stations=[[0,startFuel]]+stations+[[target,0]] # 起点 和 终点 加入 stations

        # print(stations)
        # [[0, 10], [10, 60], [20, 30], [30, 30], [60, 40], [100, 0]]

        self.cache={}
        for ele in stations:
            pos=ele[0]
            self.cache[pos]=(float('inf'),float('inf'))

        N=len(stations)

        idx=0
        pos=stations[idx][0] # 位置 0
        k=stations[idx][1] # 邮箱中的 剩余的油 数量 10

        times=self.__process(stations,N,idx,pos,k)-1 # 起点 加的油 要减去

        print(self.cache)

        if times == float('inf'):
            return -1

        return times

    def __process(self,stations,N,idx,pos,k):

        if k >= self.cache[pos][0] and self.cache[pos][1]!=float('inf'): # 不用往下搜索

            # print('({},{}) has cached'.format(pos,k))

            return float('inf') #


        else :

            if idx== N-1: # 到达 最后一个 站 即 终点

                self.cache[pos] = (k, 0)

                return 0

            min_times=float('inf')

            for i in range(idx+1,N):

                c_pos = stations[i][0]  # 到达 stations[i] 的位置
                c_k = k-(stations[i][0]-pos) # 到达 stations[i] 油箱 剩余的油量

                if c_k>=0: # 到达 stations[i] 剩余的油量 必须 >=0

                    c_k_add = c_k + stations[i][1]  # 到达 stations[i] 并加油后的 油箱的 油量
                    times=self.__process(stations,N,i,c_pos,c_k_add)
                    min_times = min(min_times, times)

                else:
                    times=float('inf')

                    # self.cache[c_pos] = (c_k,times)

                    min_times = min(min_times, times)

                    break # 后面的加油站 不用再考察了


            min_times=min_times+1

            self.cache[pos]=(k,min_times)

            return min_times

class Solution:

    # def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]])
    def solve(self, target, startFuel, stations):

        n=len(stations)

        stations=[[0,startFuel]]+stations

        # dp=np.zeros((n+1,n+1),dtype=int)
        dp=[[0 for __ in range(n+1)]for __ in range(n+1)]

        for i in range(0,n+1):
            dp[i][0]=startFuel

        for i in range(1, n + 1):
            for j in range(1,i+1):

                not_add=dp[i-1][j] # 不加油
                add=0

                if dp[i-1][j-1]>=stations[i][0]: # 能够着 加油站i
                    add=dp[i-1][j-1]+stations[i][1] # 加油

                dp[i][j]=max(not_add,add)

        # print(dp)

        for j in range( n + 1):

            if dp[n][j]>=target:
                return j

        return -1

class Test:
    def test_small_dataset(self, func):

        target = 100
        startFuel = 10
        stations = [[10, 60], [20, 30], [30, 30], [60, 40]]
        assert func(target,startFuel,stations) == 2

        target = 100
        startFuel = 1
        stations = [[10, 100]]
        assert func(target, startFuel, stations) == -1


        target=1000000
        startFuel=8663
        stations=[[31, 195796], [42904, 164171], [122849, 139112], [172890, 121724], [182747, 90912], [194124, 112994],
         [210182, 101272], [257242, 73097], [284733, 108631], [369026, 25791], [464270, 14596], [470557, 59420],
         [491647, 192483], [516972, 123213], [577532, 184184], [596589, 143624], [661564, 154130], [705234, 100816],
         [721453, 122405], [727874, 6021], [728786, 19444], [742866, 2995], [807420, 87414], [922999, 7675],
         [996060, 32691]]
        assert func(target, startFuel, stations) == 6

        target=1000
        startFuel= 299
        stations=[[13, 21], [26, 115], [100, 47], [225, 99], [299, 141], [444, 198], [608, 190], [636, 157], [647, 255],
         [841, 123]]
        assert func(target, startFuel, stations) == 4



        # TODO: 边界条件
        target = 1
        startFuel = 1
        stations = []
        assert func(target, startFuel, stations) == 0

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

        target=1000000
        startFuel=8663
        stations=[[31, 195796], [42904, 164171], [122849, 139112], [172890, 121724], [182747, 90912], [194124, 112994],
         [210182, 101272], [257242, 73097], [284733, 108631], [369026, 25791], [464270, 14596], [470557, 59420],
         [491647, 192483], [516972, 123213], [577532, 184184], [596589, 143624], [661564, 154130], [705234, 100816],
         [721453, 122405], [727874, 6021], [728786, 19444], [742866, 2995], [807420, 87414], [922999, 7675],
         [996060, 32691]] # len(stations)=25

        start = timeit.default_timer()
        print('run large dataset: N={}'.format(len(stations)))
        print(func(target,startFuel,stations))
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


if __name__ == '__main__':

    sol_d1=Solution_deprecated1()

    sol_d2 = Solution_deprecated2()

    # IDE 测试 阶段：

    target = 1000
    startFuel = 299
    stations = [[13, 21], [26, 115], [100, 47], [225, 99], [299, 141], [444, 198], [608, 190], [636, 157], [647, 255],
                [841, 123]]

    # print(sol_d1.solve(target,startFuel,stations))
    #
    # print(sol_d2.solve(target,startFuel,stations))

    sol=Solution()

    target = 100
    startFuel = 10
    stations = [[10, 60], [20, 30], [30, 30], [60, 40]]

    # print(sol.solve(target, startFuel, stations))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










