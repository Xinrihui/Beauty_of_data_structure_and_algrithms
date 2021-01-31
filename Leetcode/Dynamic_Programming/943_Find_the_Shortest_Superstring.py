#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np

import itertools as iter

class Solution:

    # def shortestSuperstring(self, A: List[str]) -> str:
    def solve(self, A):
        """


        时间复杂度 

        :param A: 
        :return: 
        """
        n=len(A)

        edges=[] # 边的集合 (源节点, 目的节点, 边长)

        A_tmp=A+[''] # 末尾 添加 1个 起点

        for pair in iter.permutations(range(n+1),2):# 长度为 2  的全排列
            edges.append( ( pair[0],pair[1],self.commonLength(A_tmp[pair[0]],A_tmp[pair[1]])) )

        # print(edges)

        # 建立 完全图
        G = {i:{} for i in range(n+1)}

        for pair in edges:
            G[pair[0]][pair[1]]=pair[2] #

        # print(G) # {0: {1: 0, 2: 1, 3: 0, 4: 3, 5: 0},
                 #  1: {0: 0, 2: 0, 3: 1, 4: 0, 5: 0},
                 #  2: {0: 0, 1: 3, 3: 0, 4: 1, 5: 0},
                 #  3: {0: 2, 1: 0, 2: 0, 4: 1, 5: 0},
                 #  4: {0: 0, 1: 1, 2: 0, 3: 0, 5: 0},
                 #  5: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}}

        end=n

        state= 2**n - 1

        start_node=(end,state)

        self.dp={}

        self.child_node={}

        d=self.__process(start_node,n,G)

        # print(self.child_node)
        # print(self.dp)

        # 追踪解
        path=[]
        node=start_node

        while node in self.child_node: # 得到 搜索路径
            path.append(node[0])
            node=self.child_node[node]

        # print(d,path)

        # 利用 搜索路径 拼接 结果字符串
        res=''
        for i in range(1,n+1): # n=5  path=[5, 2, 1, 3, 0, 4]

            prev_node = path[i-1]
            node=path[i]

            common_length=G[prev_node][node]

            res= res + A_tmp[node][common_length:]

        return res



    def __process(self, node,n,G):
        """
        旅行商问题(TSP)
        
        记忆化递归
        
        :param node: 
        :param n: 
        :param G: 
        :return: 
        """

        if node in self.dp:
            return self.dp[node]

        c=node[0]
        state=node[1]

        if state==0: # state 中没有元素了
            max_d=0
            self.child_node[node]=None # 初始状态, 没有下一个 node

        else:
            max_d=0
            for i in range(n):
                if (state>>i) & 1 ==1: # 若 第 i 位 为 1
                    next=i
                    nextState= state ^ (1<<i) # 将第 i 位 置为 0
                    nextNode=(next, nextState)

                    d= self.__process(nextNode,n,G)+G[c][next]

                    if d>= max_d:
                        max_d=d
                        self.child_node[node]=nextNode # 记录 最佳选择

        self.dp[node]=max_d

        return max_d


    def commonLength(self,s1,s2):
        """
        s1 与 s2  进行插入后 的 公共子数组的长度
        
        即 s1 的后缀 与 s2 的前缀 的最大 公共长度
        
        :param s1: 
        :param s2: 
        :return: 
        """

        n=len(s1)
        m=len(s2)

        if n <=1 or m <=1: # 长度为1 不可能 能进行插入, 因为 假设 A 中没有字符串是 A 中另一个字符串的子字符串
            return 0

        i=1

        while i < n:
            k=n-i
            if s1[i:]==s2[:k]:
                break
            i+=1
        else:
            k=0

        return k




class Test:
    def test_small_dataset(self, func):

        assert func(["catg","ctaagt","gcta","ttca","atgcatc"]) == "gctaagttcatgcatc"

        assert func(["alex","loves","leetcode"]) == "leetcodelovesalex"

        assert func(["sssv","svq","dskss","sksss"]) == "dsksssvq"

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

    # print(sol.commonLength('gcta','ctaatg'))

    # print(sol.commonLength('gctat', 'ctaatg'))

    # print(sol.commonLength('gctat', 'ttca'))

    # print(sol.commonLength('sssv', 'svq'))

    # print(sol.solve(["sssv","svq","dskss","sksss"]))

    print(sol.solve(["catg", "ctaagt", "gcta", "ttca", "atgcatc"]))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










