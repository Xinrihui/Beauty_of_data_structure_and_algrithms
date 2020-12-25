#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

class Solution:

    #  def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
    def solve(self, strs, M, N):
        """

        01 背包问题
        
        时间复杂度 O(L*M*N)

        :return: 
        """

        #0.预处理: 统计strs 中 0和1的个数
        L=len(strs)
        items=[tuple]*(L+1)

        strs=['']+strs

        for i in range(1,L+1):

            # counter=Counter(strs[i])
            count_one=0
            count_zero=0
            for char in strs[i]:
                if char=='1':
                    count_one+=1
                else:
                    count_zero+=1

            items[i]=(count_zero,count_one) #'0' 的个数, '1' 的个数

        # print(items)

        # dp=np.zeros((L+1,M+1,N+1),dtype=int) # 方便调试

        dp=[[[0 for __ in range(N+1)] for __ in range(M+1)]for __ in range(L+1)]


        for i in range(1,L+1): # 第0 行为空, 从第1行开始
            for m in range(M+1):
                for n in range(N+1):

                    case1=0 # 放入 i 物品

                    if m-items[i][0]>=0 and n-items[i][1]>=0:
                        case1= dp[i-1][m-items[i][0]][n-items[i][1]]+1

                    case2=dp[i-1][m][n] # 不放入 i 物品

                    dp[i][m][n]=max(case1,case2)

        # print(dp)

        max_num=0

        for nums in dp[L]:
            max_num=max(max_num,max(nums))

        return max_num


class Test:
    def test_small_dataset(self, func):

        strs = ["10", "0001", "111001", "1", "0"]
        m = 5
        n = 3

        assert func(strs,m,n) == 4

        strs = ["10", "0", "1"]
        m = 1
        n = 1
        assert func(strs, m, n) == 2

        strs=["00101011"]
        m=36
        n=39
        assert func(strs, m, n) == 1


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


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：
    # strs = ["10", "0001", "111001", "1", "0"]
    # m = 5
    # n = 3

    strs = ["10"]
    m = 2
    n = 2

    print(sol.solve(strs,m,n))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










