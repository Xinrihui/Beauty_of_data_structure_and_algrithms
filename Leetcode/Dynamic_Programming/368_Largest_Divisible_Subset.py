#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # largestDivisibleSubset(self, nums: List[int])
    def solve(self, nums):
        """

        时间复杂度 

        :param nums: 
        :return: 
        """
        n=len(nums)

        if n==0: # 边界处理
            return []

        sort=sorted(nums) # nums 升序排列

        # print('sort:',sort)

        dp=[0 for __ in range(n)]
        pre=[0 for __ in range(n)]

        dp[0]=1
        pre[0]=0

        for i in range(1,n):

            max_len=1
            max_idx=i # 自己 肯定可以 和自己整除

            for k in range(i-1,-1,-1):

                if sort[i]%sort[k]==0:
                    if dp[k]+1>max_len:
                        max_len=dp[k]+1
                        max_idx=k

            dp[i]=max_len
            pre[i]=max_idx

        # print('dp:',dp)
        # print('pre:',pre)

        #4. 追踪解 回溯得到解的集合

        res=[]

        c_len=max(dp)
        c_idx=dp.index(c_len)

        res.append(sort[c_idx])

        while pre[c_idx]!=c_idx:

            c_idx=pre[c_idx]
            res.append(sort[c_idx])

        return res[::-1]

class Test:
    def test_small_dataset(self, func):

        assert func([1,3,2,9,6]) == [1, 3, 6]

        assert func([1,2,4,8]) == [1,2,4,8]

        assert func([2,3,5]) == [2]

        # TODO: 边界条件
        assert func([]) == []

        assert func([4]) == [4]


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

    print(sol.solve([1,3,2,9,6]))

    # print(sol.solve([2,3,5]))

    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










