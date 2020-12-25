#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class Solution_deprecated:

    #    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
    def solve(self, nums, k):
        """
        回溯法

        时间复杂度 O(n!) 

        TLE 
        
        :param s: 
        :return: 
        """

        n=len(nums)

        sumNums=sum(nums)

        if sumNums % k !=0:
            return False

        self.target=sumNums//k

        self.S=set(range(n))


        self.k=k

        # nums=sorted(nums)

        c=None
        visit=set()
        subsum=0
        path=[]

        t=1

        self.flag=False

        self.__process(nums,c,visit,subsum,path,t)


        return self.flag

    def __process(self,nums,c,visit,subsum,path,t):

        # print('c:{},path:{}'.format(c,path))

        if len(visit)==len(nums): # 递归结束条件

            # print('subsum:{},path:{}'.format(subsum,path)) # nums 的全排列

            if t==self.k: #
                self.flag=True

            return

        if subsum == t*self.target : # 累计和 达到 t 个target ，下一个 累计和 应该为 t+1 个 target

            t+=1

        visit=visit|set([c])

        for next in self.S - visit: # nums 数组的标号

            self.__process(nums,next,visit,subsum+nums[next],path+[nums[next]],t)





class Solution:

    #  def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
    def solve(self, nums, k):
        """
        带缓存的 回溯法 (记忆化递归)


        时间复杂度 O(2^N) 

        :return: 
        """

        n = len(nums)

        sumNums = sum(nums)

        if sumNums % k != 0:
            return False

        self.target = sumNums // k

        nums=sorted(nums)

        if nums[-1] > self.target:
            return  False

        idx=n-1

        while idx >=0 and nums[idx]==self.target:
            k-=1
            idx-=1

        subset=[0]*k
        subsetList = [[] for __ in range(k)]

        res=self.__process(nums,k,subset,subsetList,idx)

        return res

    def __process(self, nums,k,subset,subsetList,idx):

        print(subsetList)

        if idx <0:
            return True

        for i in range(k):

            if subset[i]+nums[idx] <= self.target:
                subset[i]+=nums[idx]

                subsetList[i].append(nums[idx])

                if self.__process(nums,k,subset,subsetList,idx-1) == True:
                    return True

                subset[i] -= nums[idx]
                subsetList[i].pop()

        return False



class Test:
    def test_small_dataset(self, func):


        nums=[10, 10, 10, 7, 7, 7, 7, 7, 7, 6, 6, 6]
        k=3
        assert func(nums,k) == True

        nums = [4, 3, 2, 3, 5, 2, 1]
        k = 4

        assert func(nums, k) == True


        # TODO: 边界条件
        # assert func(None) == None

        # assert func(1, 0) == True

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

    def read_test_case_fromFile_list(self, dir):
        """
        解析 列表

        :param dir: 
        :return: 
        """

        with open(dir, 'r', encoding='utf-8') as file:  #

            K1 = int(file.readline().strip())
            print('K1: ', K1)

            l1 = file.readline().strip()[1:-1].split(',')

            l1 = [int(ele) for ele in l1]

            print('l1:', l1)

            return K1, l1

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
    nums = [4, 3, 5, 2, 1]
    k = 3

    print(sol.solve(nums,k))


    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










