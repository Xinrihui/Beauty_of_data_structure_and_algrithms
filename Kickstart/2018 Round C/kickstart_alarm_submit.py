#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import *

import timeit

import numba as nb


from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def process():
    """
    利用 numba 对 python 代码进行加速 到接近 C++ 的执行效率 
    
    经过试验, 对于 大数据集 N=10^6 , 执行时间为 2.3 s 相比 未加速之前的 执行时间 18.3 s
    效率有一定提升,
    
    但是 google 的平台 似乎不支持 numba , 运行报出 runtime error
    
    ref: http://numba.pydata.org/numba-doc/latest/
    
    :return: 
    """

    @nb.jit(nopython=True)
    def generate_An(N, x1, y1, C, D, E1, E2, F):

        x_prev = x1
        y_prev = y1

        # A = zeros(N + 1, dtype='uint64')  # run large dataset: time:  23.022495581 s TODO: 防溢出

        # A = zeros(N + 1, dtype='int64') # run large dataset: time:  21.125786281 s 但是还是会溢出  RuntimeWarning: overflow encountered in longlong_scalars

        A = [0] * (N + 1)

        A[1] = (x1 + y1) % F

        for i in range(2, N + 1):
            x = (C * x_prev + D * y_prev + E1) % F
            y = (D * x_prev + C * y_prev + E2) % F

            A[i] = (x + y) % F

            x_prev = x
            y_prev = y

        return A

    @nb.jit(nopython=True)
    def quick_Pow(x, n, upper_bound):
        """
        基于分治法 的快速幂乘法

        1.x^n 其中 n 必为 正整数

        2.若结果 过大, 可能会造成 后面的计算溢出(eg. 矩阵乘法 dot )
          因此, 设置上界, 对计算结果取模


        见《算法竞赛入门经典》-> 第10章 P315

        :param x: 
        :param n: 
        :param upper_bound: 

        :return: 
        """

        if n == 0:
            return 1

        sub_problem = quick_Pow(x, n // 2, upper_bound)

        ans = (sub_problem * sub_problem) % upper_bound

        if n & 1:  # n 为奇数
            ans = (ans * x) % upper_bound

        return ans

    @nb.jit(nopython=True)
    def __cal_geometric_seq_bound(q, K, upper_bound=1000000007):

        """
        等比数列求和, 公比为 q , 求和项数 n 为 self.K

        1.通过对 计算结果取模的方法 来设置上界 , 防止溢出

        2.费马小定理：若 p是质数，且a、p互质，那么a^(p-1) mod p = 1。

        现在，我们要求 a/c mod p，通过一系列转化，除法就会神奇地消失...

        a / c mod p
        = a / c mod p * 1
        = a / c mod p * c^(p-1) mod p
        = a * c^(p-2) mod p

        ref: https://me.csdn.net/xiaogengyi

        :param q: 
        :return: 
        """
        res = 0
        # n = self.K

        if q == 1:

            res = K

        elif q > 1:

            # res = (self.quick_Pow(q, n + 1, self.upper_bound) - q) // (q - 1) # 整数除法 // : 等比数列中的元素 都是整数, 因此求和后肯定还是整数

            a = q * (quick_Pow(q, K, upper_bound) - 1) % upper_bound

            res = (a * quick_Pow(q - 1, upper_bound - 2, upper_bound)) % upper_bound

        return res

    @nb.jit(nopython=True)
    def solve(N, K, A):
        """

        优化思路见  def solve_deprecated 的 函数注释

        :param N: 
        :param K: 
        :param A: 
        :return: 
        """

        upper_bound = 1000000007

        result = 0
        last_sum = 0

        for i in range(1, N + 1):
            # print(j)

            last_sum += __cal_geometric_seq_bound(i, K, upper_bound)

            last_sum = last_sum % upper_bound  # TODO: 每一步计算 都 加上取模, 防止溢出

            tmp = (last_sum * (N - i + 1)) % upper_bound

            result += (A[i] * tmp) % upper_bound

            result = result % upper_bound

        return result

    #     N, K, x1, y1, C, D, E1, E2, F=2, 3, 1, 2, 1, 2, 1, 1, 9
    #     A=generate_An(N, x1, y1, C, D, E1, E2, F)
    #     __cal_geometric_seq_bound(10,20)
    #     print(solve(N, K, A))

    # N, K, x1, y1, C, D, E1, E2, F = 10, 10, 10001, 10002, 10003, 10004, 10005, 10006, 89273
    # A = generate_An(N, x1, y1, C, D, E1, E2, F)
    # print(solve(N, K, A))

    x1, y1, C, D, E1, E2, F = 10001, 10002, 10003, 10004, 10005, 10006, 89273
    N = int(pow(10, 6))
    K = int(pow(10, 4))
    A = generate_An(N, x1, y1, C, D, E1, E2, F)

    start = timeit.default_timer()
    print('run large dataset: ')
    print(solve(N, K, A))
    end = timeit.default_timer()
    print('time: ', end - start, 's')



    # T = int(input().strip())  # 一共T个 测试数据
    #
    # for t in range(1, T + 1):
    #     N, K, x1, y1, C, D, E1, E2, F = [int(i) for i in input().split(' ')]
    #
    #     A = generate_An(N, x1, y1, C, D, E1, E2, F)
    #
    #     res = solve(N, K, A)
    #
    #     print('Case #{}: {}'.format(t, res))


process()