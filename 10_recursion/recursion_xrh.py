#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit
import numpy as np

import math

import random as rand

class solutions:

    def take_stairs(self,n):
        """
        走楼梯问题：
        假如这里有 n 个台阶，每次你可以跨 1 个台阶或者 2 个台阶，请问走这 n 个台阶有多少种走法？
        
        递归解：
        
        1. 递推公式
        f(n)=f(n-1)+f(n-2)
        
        2.递归终止条件 
        f(1)=1
        f(2)=1
        
        ref: https://time.geekbang.org/column/article/41440
        :param n: 
        :return: 
        """
        if n==2 or n==1:
            return 1

        return self.take_stairs_recursion(n-1)+self.take_stairs_recursion(n-2)

    def take_stairs_v2(self, n):
        """
        走楼梯问题 v2
        1.递归解，并加入 缓存机制，解决重复子问题
        
        :param n: 
        :return: 
        """
        self.cache={}

        return self.__process(n)

    def __process(self,n):

        if n==2 or n==1:
            return 1

        if self.cache.get(n-1)==None:
            left=self.__process(n-1)

        else:
            left=self.cache[n-1]

        if self.cache.get(n - 2) == None:
            right = self.__process(n - 2)

        else:
            right = self.cache[n - 2]

        return left+right

    def take_stairs_v3(self,n):
        """
        走楼梯问题 v3
        
        动态规划解
        1.条件：
        （1）重复子问题
        （2）最优子结构 
    
        :param n: 
        :return: 
        """

        dp=np.zeros(n+1,dtype=int)

        dp[1]=1
        dp[2]=1

        for i in range(3,n+1):
            dp[i]=dp[i-1]+dp[i-2]

        return dp[n]

    def take_stairs_v4(self,n):
        """
        走楼梯问题 v4
        
        动态规划解
        1.优化空间复杂度
        
        :param n: 
        :return: 
        """

        if n<=2:
            return 1

        prev=1
        pre_prev=1

        for i in range(3, n + 1):
            current=prev+pre_prev

            pre_prev=prev
            prev = current

        return current


if __name__ == '__main__':
    sol = solutions()

    print(sol.take_stairs_v4(6))


