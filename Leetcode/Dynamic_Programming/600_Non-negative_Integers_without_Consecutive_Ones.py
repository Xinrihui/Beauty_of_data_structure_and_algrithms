#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

import math

class Solution_deprecated:

    # def findIntegers(self, num: int) -> int:
    def solve(self, num):
        """

        时间复杂度 
        
         1 <= num <= 10^9
        
        :param num: 
        :return: 
        """

        if num==1:
            return 2

        L= int(math.log2(num))

        dp0=[0]*(L+1)
        dp1 = [0] * (L + 1)

        dp0[1]= 1
        dp1[1] = 1

        for i in range(2,L+1):

            dp0[i]=dp0[i-1]+dp1[i-1]
            dp1[i]=dp0[i-1]

        Num=dp0[L]+dp1[L] # 长度为 L 的序列, 不包含连续的 1 的组合的数目

        start= 2**L
        end=num

        Num2=0

        for ele in range(start,end+1):

            if self.check_faster(ele)==False:
                Num2+=1

        return Num+Num2

    def check_faster(self, n):
        """
        判断 n 是否 存在连续的 位为 1

        bin(6)= 110  True
        bin(5)= 101  False

        :param n: 
        :return: 
        """

        return  ( n & (n >> 1) ) > 0


    def check(self,n):
        """
        判断 n 是否 存在连续的 位为 1
        
        bin(6)= 110  True
        
        :param n: 
        :return: 
        """

        bits_num= len(bin(n))-2 # 二进制的位数

        flag=0 # 为1的标记位

        for i in range(bits_num):
            if n & (1<<i) > 0 : # n 的第i位为1
                if flag ==1: # 前一位 已经为1 , 当前 位还是 1 , 说明连续出现了1

                    return True
                else:
                    flag=1
            else:
                flag=0

        return False


class Solution:
    # def findIntegers(self, num: int) -> int:
    def solve(self, num):
        """

        时间复杂度 

         1 <= num <= 10^9

        :param num: 
        :return: 
        """

        if num == 1:
            return 2

        L = int(math.log2(num))

        dp0 = [0] * (L + 1)
        dp1 = [0] * (L + 1)
        dp = [0] * (L + 1)

        dp[0]=1

        dp0[1] = 1
        dp1[1] = 1
        dp[1] = 2

        for i in range(2, L + 1):
            dp0[i] = dp0[i - 1] + dp1[i - 1]
            dp1[i] = dp0[i - 1]

            dp[i] = dp0[i]+dp1[i] # 长度为 i 的序列, 不包含连续的 1 的组合的数目

        Num=0

        bits_num=len(bin(num))-2 # 二进制的位数

        flag=False
        for i in range(bits_num-1,-1,-1):

            if (num >>i) & 1 == 1: #  num 的 第i 位为1
                Num+=dp[i]

                if flag==True: # 出现连续的1, 跳出循环
                    break

                flag=True

            else:
                flag=False # 第i 位为1 的标记


        if self.check_faster(num)==False:
            Num+=1


        return Num

    def check_faster(self, n):
        """
        判断 n 是否 存在连续的 位为 1

        bin(6)= 110  True
        bin(5)= 101  False

        :param n: 
        :return: 
        """

        return  ( n & (n >> 1) ) > 0



class Test:
    def test_small_dataset(self, func):

        assert func(8) == 6

        assert func(10000000) == 103682

        assert func(100000000) == 514229

        assert func(1000000000) == 2178309

        # TODO: 边界条件
        assert func(1) == 2



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

    # sol = Solution_deprecated()

    sol = Solution()

    # IDE 测试 阶段：

    # print(sol.solve(8))

    print(sol.solve(10000000))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










