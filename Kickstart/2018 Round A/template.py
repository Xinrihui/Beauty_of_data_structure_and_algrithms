#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import *

import timeit

class Test:

    def test_small_dataset(self,func):

        assert  func()==''

        assert  func()==''

        assert  func()==''


    def test_large_dataset(self,func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题
        
        Limits
        
        
        :param func: 
        :return: 
        """

        N = int(2 * pow(10, 4))
        max_v= int(pow(10,9))

        l = random.randint(max_v, size=N)
        l1 = list(l)

        start = timeit.default_timer()
        print('run large dataset: ')
        func( )
        end = timeit.default_timer()
        print('time: ', end - start, 's')



class solutions:
    """
   
    """

    def _f(self, values, threshold):

        pass


    def xrh_naive(self, N, K, values):

        pass


    def xrh(self, N, K, values):

        pass


if __name__ == '__main__':

    sol=solutions()

    # sol.xrh()
    # sol.xrh()

    # IDE 测试 阶段：
    test=Test()
    test.test_small_dataset(sol.xrh)

    test.test_large_dataset(sol.xrh)

    # 提交 阶段：
    # pycharm 开命令行 提交
    #E:\python package\python-project\Beauty_of_data_structure_and_algrithms\kickstart\2018 Round A>python Scrambled_Words.py < inputs
    # Case #1: 4


    T = int(input()) # 一共T个 测试数据

    for t in range(1, T + 1):

        N_K = [int(i) for i in input().split(' ')]

        N,K=N_K[0],N_K[1]

        values=[int(i) for i in input().split(' ')]

        res = sol.xrh(N,K,values)

        print('Case #{}: {}'.format(t, res))









