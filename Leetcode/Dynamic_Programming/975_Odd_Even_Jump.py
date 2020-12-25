#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
from sortedcontainers import *

class Solution:

    # oddEvenJumps(self, A: List[int]) -> int:
    def solve(self, A):
        """

        时间复杂度 

        :param A: 
        :return: 
        """
        n=len(A)

        # dp=np.zeros((n,2),dtype=bool)

        dp= [ [False,False] for __ in range(n)]

        dp[n-1][1]=True # 奇跳
        dp[n - 1][0] = True # 偶跳

        sk = SortedKeyList([(A[n-1], n-1)], key=lambda ele: ele[0]) # 指定 参与比较的key

        print(sk)

        for i in range(n-2,-1,-1):

            heigh_idx =sk.bisect_key_right(A[i]) # 二分查找 右边插入的位置
            low_idx = sk.bisect_key_left(A[i]) # 二分查找 左边 插入位置

            if heigh_idx==low_idx: # A[i] 在 sk 中不存在

                if heigh_idx > len(sk)-1:
                    #  A[i] 比 sk 中的最大的元素还大, 无法向上跳了
                    dp[i][1] = False

                elif sk[heigh_idx][0]>A[i]:
                    # 奇跳 向上跳
                    dp[i][1]= dp[ sk[heigh_idx][1] ][0]

                low_idx=low_idx-1 # 查找的键不存在时，bisect_key_left 返回的位置是 右边的插入位置

                if low_idx == -1:
                    #  A[i] 比 sk 中的最小的元素还小, 无法向下跳了
                    dp[i][0] = False

                elif sk[low_idx][0]<A[i]:
                    # 偶跳 向下跳
                    dp[i][0]= dp[sk[low_idx][1] ][1]

                sk.add((A[i], i))

            elif heigh_idx==low_idx+1: # A[i] 在 sk 中 存在, bisect_key_left 返回的是左边插入的位置

                # 奇跳 向上跳
                dp[i][1] = dp[ sk[low_idx][1] ][0]
                # 偶数跳 向下跳
                dp[i][0] = dp[ sk[low_idx][1] ][1]

                sk.pop(low_idx) # 先删 再加 等价于 更新
                sk.add((A[i], i))


            print(sk)

        print(dp)

        count=0
        for ele in dp:
            if ele[1]==True:
                count+=1

        return count

class Test:
    def test_small_dataset(self, func):

        assert func([1,2,3,2,1,4,4,5]) == 6

        assert func([2,3,1,1,4]) == 3

        assert func([5,1,3,4,2]) == 3


        # TODO: 边界条件
        assert func([2]) == 1



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

    print(sol.solve([5,1,3,4,2]))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










