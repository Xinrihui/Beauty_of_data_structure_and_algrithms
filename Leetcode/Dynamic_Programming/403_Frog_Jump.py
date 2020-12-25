#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    #  canCross(self, stones: List[int]) -> bool:
    def solve(self, stones):
        """

        时间复杂度 

        :param stones: 
        :return: 
        """
        L = len(stones)

        if L==2: # 边界情况
            return stones[1]-stones[0]==1

        pos_list=[0]+stones

        setPos={}

        for i in range(L+1):
            setPos[pos_list[i]]=i

        # print('setPos:',setPos)

        self.cache={}

        self.flag=False # 找到 可到达路径 的标记

        res=False
        c = L

        for i in range(c-1,1,-1):

            k=pos_list[c]-pos_list[i]

            r=self.__process(setPos,pos_list,i,k) # TODO: 少用全局变量,不变的变量(setPos,pos_list)用参数 传入

            self.cache[(i, k)] = r

            res= res|r

        return res

    def __process(self,setPos,pos_list,c,K):

        if (c,K) not in self.cache:

            if c==2:
                if 1 in {K-1,K,K+1}:
                    self.flag=True  # 找到了一条 可到达路径

                    return True

                else:
                    return False

            if self.flag==False: # 尚未找到 可到达路径

                diff=[K-1,K,K+1]
                res=False

                for d in diff:
                    prev=pos_list[c]-d

                    if prev in setPos:
                        i=setPos[prev]
                        if i<c:
                            r=self.__process(setPos,pos_list,i,d)
                            self.cache[(i, d)]=r

                            res= res|r

                return res

            else: # 剪枝: 已经找到 可到达路径, 结束子问题的搜索
                return True


        else: # 子问题 已经被计算, 直接返回 缓存结果
            return self.cache[(c,K)]


class Test:
    def test_small_dataset(self, func):

        assert func([0,1,3,5,6,8,12,17]) == True

        assert func([0,1,2,3,4,8,9,11]) == False

        assert func([0,1,3,5])== True

        assert func([0, 1, 3]) == True


        # TODO: 边界条件
        assert func([0,1]) == True

        assert func([0,2]) == False



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

    print(sol.solve([0,1,3,5,6,8,12,17]))

    # print(sol.solve([0,1]))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










