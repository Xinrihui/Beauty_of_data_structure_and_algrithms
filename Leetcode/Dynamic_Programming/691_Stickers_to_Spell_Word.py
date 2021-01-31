#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # def minStickers(self, stickers: List[str], target: str) -> int:
    def solve(self, stickers,target):
        """
        
        时间复杂度 
        记忆化递归 
        ( 多重背包 )
        

        :param s: 
        :return: 
        """

        target_dic=Counter(target)
        target_list= target_dic.items()

        N=len(target_list) # 物品种类

        # print(target_list)

        key_idx={}

        need=[0]*N

        for ele in target_list:

            key=ele[0]
            time=ele[1]

            idx=len(key_idx)
            key_idx[key]=idx

            need[idx]=time

        # print(key_idx)
        # print(need)

        item_list=[]

        for word in stickers:

            weight=[0]*N

            for ele in Counter(word).items():
                key = ele[0]
                time = ele[1]

                if key in key_idx:
                    weight[ key_idx[key] ] = time

            item_list.append(weight)

        # print(item_list)
        self.dp={}

        res=self.__process(N,item_list,tuple(need))

        if res==float('inf'):
            res=-1

        return res

    def __process(self, N,bag_list,need):
        """
        
        :param N: 
        :param bag_list: 
        :param need: 
        :return: 
        """
        if need in self.dp:
            return self.dp[need]

        # 递归结束条件
        if max(need) ==0: # 没有物品需要被放入

            self.dp[need]=0 # 初始化 状态

            return 0

        min_Num= float('inf') # 放入 的 最少 礼包 个数

        for bag in bag_list: # 选择 一个 礼包

            diff = [0]*N

            for i in range(N): # 礼包 中的 每一个 物品的个数

                item=need[i]-bag[i]
                if item<0: # 所需的 不能为负值
                    item=0
                diff[i]=item

            diff=tuple(diff)

            if diff != need: # 添加了物品 要有变化

                min_Num = min( min_Num, self.__process(N,bag_list,diff) + 1 )

        self.dp[need]=min_Num

        return min_Num



class Test:
    def test_small_dataset(self, func):

        assert func(["with", "example", "science"], "thehat") == 3

        assert func(["notice", "possible"], "basicbasic") == -1

        # assert func("cbbd") == 'bb'
        #
        # assert func("cbbc") == 'cbbc'
        #
        # assert func("abcd") == 'a'

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

    print(sol.solve(["with", "example", "science"], "thehat"))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










