#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import *

import timeit


class Test:
    def test_small_dataset(self, func):

        N, K, x1, y1, C, D, E1, E2, F = 2, 3, 1, 2, 1, 2, 1, 1, 9
        A = sol.generate_An(N, x1, y1, C, D, E1, E2, F)

        print(func(N, K, A))

        assert func(N, K, A) == 52

        N, K, x1, y1, C, D, E1, E2, F = 10, 10, 10001, 10002, 10003, 10004, 10005, 10006, 89273
        A = sol.generate_An(N, x1, y1, C, D, E1, E2, F)

        print(func(N, K, A))

        assert func(N, K, A) == 739786670

        print('test small dataset success!')

    def test_large_dataset(self, func):
        """
        自己 生成大的 数据集，
        (1) 查看算法的运行时间，解决 TTL(超时) 问题
        (2) 查看 算法的 内存占用 解决 OOM 问题

        Limits

        1 ≤ x1 ≤ 10^5.
        1 ≤ y1 ≤ 10^5
        1 ≤ C ≤ 10^5.
        1 ≤ D ≤ 10^5.
        1 ≤ E1 ≤ 10^5.
        1 ≤ E2 ≤ 10^5.
        1 ≤ F ≤ 10^5.

        1 ≤ N ≤ 10^6.
        1 ≤ K ≤ 10^4.

        :param func: 
        :return: 
        """

        x1 = int(pow(10, 5))
        y1 = int(pow(10, 5))

        C = int(pow(10, 5))
        D = int(pow(10, 5))

        E1 = int(pow(10, 5))
        E2 = int(pow(10, 5))
        F = int(pow(10, 5))

        # x1, y1, C, D, E1, E2, F = 10001, 10002, 10003, 10004, 10005, 10006, 89273

        # N = int(pow(10, 4)) # 14.281854638999999 s
        # K = int(pow(10, 4))

        # N = int(pow(10, 5)) # 208.118737882 s
        # K = int(pow(10, 4))

        # N = 100
        # K = 20

        N = int(pow(10, 6))
        K = int(pow(10, 4))

        A = sol.generate_An(N, x1, y1, C, D, E1, E2, F)

        start = timeit.default_timer()
        print('run large dataset: ')
        func(N, K, A)
        end = timeit.default_timer()
        print('time: ', end - start, 's')


class solutions:
    """

    利用 等比数列求和 
    将 i=1 ,i=2 ,... i=K 一共 K 步骤的运算 合并为 1 步

    """

    def generate_An(self, N, x1, y1, C, D, E1, E2, F):

        A = []

        x_prev = x1
        y_prev = y1

        A.append((x1 + y1) % F)

        for i in range(2, N + 1):
            x = (C * x_prev + D * y_prev + E1) % F
            y = (D * x_prev + C * y_prev + E2) % F

            A.append((x + y) % F)

            x_prev = x
            y_prev = y

        return A

    def __conti_subarray(self, nums):
        """
        生成 目标数组 的连续子数组

        eg. nums=[1,4,2]

        res=[ [1],[4],[2],[1,4].[4,2],[1,4,2] ]

        生成 子数组 的时间复杂度是 O(n^2) n 为目标数组的长度

        当 n = 10^6 , n^2=10^12 单机无法 在 1min 内计算完

        :param nums: 
        :return: 
        """

        res = []

        for length in range(1, len(nums) + 1):

            i = 0
            while (i + length) <= len(nums):
                res.append(nums[i:i + length])  # TODO: len(nums)=10^6 , 此步骤执行的次数为 10^12 凉凉
                i += 1

        return res

    def __cal_geometric_seq(self, q):
        """
        等比数列求和, 公比为 q , 求和项数为 self.K
        :param q: 
        :return: 
        """

        if q not in self.cache_sum:

            res = 0
            n = self.K

            if q == 1:

                res = n

            elif q > 1:

                res = (q - q ** (n + 1)) // (1 - q)  # 整数除法 // : 等比数列中的元素 都是整数, 因此求和后肯定还是整数

            self.cache_sum[q] = res  # 计算结果缓存

        else:
            res = self.cache_sum[q]

        return res

    def __cal_subA_sum(self, sub_A):
        """
        对子数组 计算求和

        eg. sub_A=[3,2]

        res=3*(1+1+1) + 2*(2^1 +2^2+2^3)

        :param sub_A: 
        :return: 
        """
        # 利用矩阵运算, 加速计算

        # sub_A = array(sub_A) #TODO: dot(x,y) 默认 使用 int32 进行计算, 会发生溢出

        sub_A = array(sub_A, dtype='uint64')  # 指定数据类型, 避免发生溢出

        sum_seq = array(list(map(lambda x: self.__cal_geometric_seq(x), range(1, len(sub_A) + 1))), dtype='uint64')

        # print(sum_seq)

        return dot(sub_A, sum_seq)

    def solve_naive(self, N, K, A):
        pass

    def solve(self, N, K, A):

        # 1. 生成所有的连续子数组

        sub_A_list = self.__conti_subarray(A)  # TODO: 时间复杂度 过高

        # print(sub_A_list) #[[3], [2], [3, 2]]

        # 2.对每一个子数组 进行求和

        self.cache_sum = {}  # 对等比数列的求和结果缓存
        self.N = N
        self.K = K

        res_sum = 0

        for sub_A in sub_A_list:
            res_sum += self.__cal_subA_sum(sub_A)

        return "{:.0f}".format(
            res_sum % 1000000007)  # TODO: 注意看 题目的 output 条件: Since POWER could be huge, print it modulo 1000000007 (109 + 7).



class solutions_opt:
    """

    1.利用 等比数列求和 将 i=1 ,i=2 ,... i=K 一共 K 步骤的运算 合并为 1 步

    2.由 class solutions 发现 生成目标数组的 连续子数组的 时间复杂度为 O(n^2) 而 目标数组的长度上限为 10^6 ,
      10^12 的数量级 导致程序超时, 并且 生成的子数组序列 占用的内存 远超过 1GB;

      本题目 最终 要求 所有 连续子数组的 指数幂 的和, 所以尝试找到 其中的规律, 先做合并同类项 再整体求和 , 
      可以利用递推公式, 算法的时间复杂为 O(n) 

    """

    def generate_An(self, N, x1, y1, C, D, E1, E2, F):

        x_prev = x1
        y_prev = y1

        # A = zeros(N + 1, dtype='uint64')  # run large dataset: time:  23.022495581 s TODO: 防溢出
        # A = zeros(N + 1, dtype='int64') # run large dataset: time:  21.125786281 s 但是还是会溢出  RuntimeWarning: overflow encountered in longlong_scalars

        A=[0]*(N+1)

        A[1] = (x1 + y1) % F

        for i in range(2, N+1 ):
            x = (C * x_prev + D * y_prev + E1) % F
            y = (D * x_prev + C * y_prev + E2) % F

            A[i] = (x + y) % F

            x_prev = x
            y_prev = y

        return A

    def quick_Pow(self, x, n, upper_bound):
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

        sub_problem = self.quick_Pow(x, n // 2, upper_bound)

        ans = (sub_problem * sub_problem) % upper_bound

        if n & 1:  # n 为奇数
            ans = (ans * x) % upper_bound

        return ans

    def __cal_geometric_seq(self, q):
        """
        等比数列求和, 公比为 q , 求和项数为 self.K
        :param q: 
        :return: 
        """

        res = 0
        n = self.K

        if q == 1:

            res = n

        elif q > 1:

            res = (q ** (n + 1) - q) // (q - 1)  # 整数除法 // : 等比数列中的元素 都是整数, 因此求和后肯定还是整数

        return res

    def __cal_geometric_seq_bound(self, q):

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

            res = self.K

        elif q > 1:

            # res = (self.quick_Pow(q, n + 1, self.upper_bound) - q) // (q - 1) # 整数除法 // : 等比数列中的元素 都是整数, 因此求和后肯定还是整数

            a = q * (self.quick_Pow(q, self.K, self.upper_bound) - 1) % self.upper_bound

            res = (a * self.quick_Pow(q - 1, self.upper_bound - 2, self.upper_bound)) % self.upper_bound

        return res

    def solve(self, N, K, A):
        """

        优化思路见  def solve_deprecated 的 函数注释

        :param N: 
        :param K: 
        :param A: 
        :return: 
        """

        self.K = K

        self.upper_bound = 1000000007

        result = 0
        last_sum = 0

        for i in range(1, N + 1):
            # print(j)

            last_sum += self.__cal_geometric_seq_bound(i)

            last_sum = last_sum % self.upper_bound  # TODO: 所有能取模的地方都 加上取模

            tmp = (last_sum * (N - i + 1)) % self.upper_bound

            result += ( A[i] * tmp ) % self.upper_bound

            result = result % self.upper_bound

        return result

    def solve_deprecated(self, N, K, A):
        """
        大数据集 还是 TLE ( 我要哭了 == )

        算法时间复杂度: O(N)

        1. N=10w 量级 (10^6) , 单纯 for 循环遍历一遍的时间:  0.106 s 


        测试 大数据集 的实际耗时

        N=1w
        N = int(pow(10, 5)) # 208.118737882 s

        算法的实际耗时大大超过估算, 经分析, 慢在 等比数列求和函数 def __cal_geometric_seq()
        在 该函数中, 显然 耗时最大的为 计算指数 步骤 pow(q,n)

        最大情况: q=N=10^6 , n=K=10^4        
        pow(10^6,10^4)= 10^(60000)  远远超过 int64 所能表示的范围, 
        但是由于 python 中的整形可以无限大, 所以结果可以正确计算 而不会发生溢出, 但是 这也大大降低了效率, 导致 TLE

        2. 注意到 题中要求 对 计算结果 取模后  再进行 output, 也是 考虑到 计算结果 huge ;

           由于对结果取模, 所以 对中间 计算过程 的结果 取模后 再进行后续的计算 并不会改变最后的结果( 除法例外 )

           eg. 
           A=16 B=14  A*B % 10 = (A%10)*(B%10)
                      A+B % 10 = (A%10)+(B%10) = A+(B%10)

           因此, 我们可以对其中间步骤的中间结果取模 (eg. 计算指数 pow(q,n) 时, 考虑 利用分治法求指数 ), 这样即可以防止溢出, 又可以减少运行时间

        3. 上述优化过程 的实现见 def solve()

        :param N: 
        :param K: 
        :param A: 
        :return: 
        """

        self.K = K

        res_sum = 0

        prev_S = 0

        for j in range(N):
            print(j)

            S = prev_S + self.__cal_geometric_seq(j + 1)

            res_sum += A[j] * (N - j) * S

            prev_S = S

        return "{:.0f}".format(
            res_sum % 1000000007)  # TODO: 注意看 题目的 output 条件: Since POWER could be huge, print it modulo 1000000007 (109 + 7).


if __name__ == '__main__':

    sol = solutions_opt()

    flag = 1  # TODO:提交代码记得修改 状态

    # IDE 测试 阶段：
    if flag == 1:

        # N, K, x1, y1, C, D, E1, E2, F=2, 3, 1, 2, 1, 2, 1, 1, 9
        # A=sol.generate_An(N, x1, y1, C, D, E1, E2, F)

        N, K = 3, 2
        A = [1, 4, 2]
        # print(sol.solve(N, K, A))


        # IDE 测试 阶段：
        test = Test()
        test.test_small_dataset(sol.solve)

        test.test_large_dataset(sol.solve)


    # 提交 阶段：
    elif flag == 2:
        # pycharm 开命令行 提交
        # E:\python package\python-project\Beauty_of_data_structure_and_algrithms\kickstart\2018 Round A>python Scrambled_Words.py < inputs
        # Case #1: 4

        T = int(input().strip())  # 一共T个 测试数据

        for t in range(1, T + 1):
            N, K, x1, y1, C, D, E1, E2, F = [int(i) for i in input().split(' ')]

            A = sol.generate_An(N, x1, y1, C, D, E1, E2, F)

            res = sol.solve(N, K, A)

            print('Case #{}: {}'.format(t, res))




