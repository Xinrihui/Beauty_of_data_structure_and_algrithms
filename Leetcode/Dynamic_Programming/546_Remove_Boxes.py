#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # removeBoxes(self, boxes: List[int]) -> int:
    def solve_dp(self, boxes):
        """
        升维 动态规划
        
        时间复杂度 O(n^4)
        
        TLE 
        
        :param boxes: 
        :return: 
        """

        n=len(boxes)

        boxes = [0] + boxes

        # dp=np.zeros((n+1,n+1,n+1),dtype=int)
        dp = [[[0 for __ in range(n+1)] for __ in range(n+1)]for __ in range(n+1)]

        for l in range(1,n+1):
            for i in range(1,n-l+2):
                for k in range(0,n-i+1):
                    j=i+l-1

                    # print('i:{},j:{},k:{}'.format(i,j,k))

                    a=0
                    if i <= j-1:
                        a=dp[i][j-1][0]

                    case1=a+(k+1)**2

                    case2=0
                    for p in range(i,j):

                        if boxes[p]==boxes[j]:

                            c=0
                            if i <= p:
                                c=dp[i][p][k+1]
                            d=0
                            if p+1 <= j-1:
                                d=dp[p+1][j-1][0]

                            # print('p:{},c:{},d:{}'.format(p,c,d))

                            case2=max(case2,c+d)

                    # print('case1:{},case2:{}'.format(case1,case2))

                    dp[i][j][k] = max(case1, case2)



        # print(dp)

        res=dp[1][n][0]

        return res



    # removeBoxes(self, boxes: List[int]) -> int:
    def solve_backtracking(self, boxes):
        """
        带缓存 的回溯法 (记忆化递归)     
           
        1 <= boxes.length <= 100
        1 <= boxes[i] <= 100
        
        自顶向下递归
        
        时间复杂度 O(n^4)

        :param boxes: 
        :return: 
        """

        n=len(boxes)

        boxes=[0]+boxes

        # self.dp=np.zeros((n+1,n+1,n+1),dtype=int)
        self.dp = [[[0 for __ in range(n+1)] for __ in range(n+1)]for __ in range(n+1)]

        l=1
        r=n
        k=0

        score=self.__process(boxes,l,r,k)

        return score


    def __process(self,boxes,l,r,k):
        """
        递归 求解
        
        :param boxes: 
        :param l: 
        :param r: 
        :return: 
        """

        if l>r: # 结束递归 条件
            return 0

        if self.dp[l][r][k]>0:
            return self.dp[l][r][k]

        # 剪枝 优化
        # 对于 连续 相同颜色的盒子, 不用从头到尾一个一个 找切分点了
        # eg.
        # boxes[3:5+1]==[2,2,2]
        # score= dp[3][3][2]+dp[4][3][0]

        while l<r-1 and boxes[r-1]==boxes[r]:
            r-=1
            k+=1

        s_case1= self.__process(boxes,l,r-1,0)+ (k+1)**2

        s_case2=0 # 分数最小为0

        for p in range(l,r):

            if boxes[p]==boxes[r]:
                s_case2=max(s_case2,self.__process(boxes,l,p,k+1)+self.__process(boxes,p+1,r-1,0))

        self.dp[l][r][k]=max(s_case1,s_case2)

        return self.dp[l][r][k]





class Test:
    def test_small_dataset(self, func):

        assert func([1, 3, 2, 2, 2, 3, 4, 3, 1]) == 23

        assert func([2,2,1,2,2]) == 17

        assert func([1,2,1,3,1]) == 11

        # TODO: 边界条件
        # assert func(None) == None

        assert func([1]) == 1


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

        l=[57, 19, 24, 72, 53, 39, 47, 70, 48, 87, 69, 53, 73, 19, 50, 87, 41, 17, 55, 13, 94, 80, 83, 5, 32, 17, 53, 86,
         77, 34, 51, 85, 2, 1, 48, 74, 27, 41, 70, 15, 13, 29, 86, 22, 92, 75, 85, 56, 76, 82, 44, 30, 16, 7, 38, 6, 20,
         24, 13, 31, 38, 2, 97, 42, 69, 98, 24, 97, 12, 76, 98, 30, 37, 23, 72, 85, 16, 72, 12, 29, 9, 15, 85, 15, 57,
         54, 40, 20, 33, 76, 92, 44, 29, 59, 18, 7, 53, 43, 37, 12]

        start = timeit.default_timer()
        print('run large dataset: ')
        func(l)
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

    sol = Solution()

    # IDE 测试 阶段：
    boxes = [1, 3, 2, 2, 2, 3, 4, 3, 1]

    # boxes = [2, 2, 2]

    # print(sol.solve(boxes))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve_backtracking)

    # test.test_large_dataset(sol.solve_dp)

    test.test_large_dataset(sol.solve_backtracking)










