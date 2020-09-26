#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

import itertools

class Test:
    def test_small_dataset(self, func):

        assert func() == ''

        assert func() == ''

        assert func() == ''

    def test_large_dataset(self, func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题

        Limits


        :param func: 
        :return: 
        """

        N = int(2 * pow(10, 4))
        max_v = int(pow(10, 9))


        start = timeit.default_timer()
        print('run large dataset: ')
        func()
        end = timeit.default_timer()
        print('time: ', end - start, 's')


class IntDigit:
    """
    by XRH 
    date: 2020-06-25

    不用 将整数 转换为 字符数组，而用 纯数学的方法 拿到 整数中的 各个位(digit)

    """

    @staticmethod
    def getLength(n):
        """
        返回 整数 的位数

        ref: https://stackoverflow.com/questions/2189800/length-of-an-integer-in-python

        :param n: 
        :return: 
        """
        import math

        return int(math.log10(n)) + 1

    @staticmethod
    def getMinDigit(n):
        """
        返回 整数的 最低位
        :param n: 
        :return: 
        """
        return n % 10

    @staticmethod
    def getMaxDigit(n):
        """
        返回 整数的 最高位

        :param n: 
        :return: 
        """

        while n >= 10:
            n = n // 10

        return n

    @staticmethod
    def IntToList(n):
        """
        从高位到 低位 返回 整数 的 每一位 

        注意此函数 的时间复杂度是 O(length(n))  
        eg. 
        n=849
        res=[8,4,9]

        :param n: 
        :return: 
        """

        digit_list = []

        while n > 0:
            digit = n % 10

            digit_list.append(digit)

            n = n // 10

        return digit_list[::-1]

    @staticmethod
    def IntHexConversion(n, x):
        """
        进制转换 

        1.将 整数(10进制) n 转换为  x 进制表示 ,并以列表的形式输出 

        2. 利用除 x 取余数法

        3.注意此函数 的时间复杂度是 O(length(n))  

        eg. 
        n=6
        x=2 
        digit_list=[1,1,0]

        n=5
        x=2 
        digit_list=[1,0,1]

        :param n: 
        :return: digit_list
        """

        digit_list = []

        while n > 0:
            digit = n % x

            digit_list.append(digit)

            n = n // x

        return digit_list[::-1]

    @staticmethod
    def ListToInt(digit_list):
        """
        将 digit_list 转换回 整数

        eg. 
        digit_list=[8,4,9]
        res=849

        digit_list=[0,1,0]
        res=10

        digit_list=[0,0]
        res=0

        digit_list=[]
        res=0

        :param digit_list: 
        :return: 
        """
        import functools

        return functools.reduce(lambda total, d: 10 * total + d, digit_list, 0)

    @staticmethod
    def getNthDigit(num, N):
        """
        返回 整数 从低到高 的第 N 位

        num=987654321 N=0 res=1


        ref: https://stackoverflow.com/questions/39644638/how-to-take-the-nth-digit-of-a-number-in-python
        :param num: 
        :param N: 
        :return: 
        """

        return num // 10 ** N % 10


class solutions:
    """
    输入正整数n，从小到大输出所有形如 abcde/fghij=n的表达式，其中a-j恰好为数字0-9的一个排列（可以有前导0），
    
    2=<n<=79.

    """

    def solve(self, n):
        """

        :param n: 
        :return: 
        """


        num_set = set([str(i) for i in range(10)])  # 可选数字的集合 , 用字符表示

        res_list = []

        for mu_list in itertools.permutations(num_set, 5):  # 0-9 的长度是 5 的全部的排列

            mu_list=list(mu_list) # 一定是 5 位
            mu_set=set(mu_list)

            mu = int(''.join(mu_list)) # 分母

            # zi_num_set = num_set - set(mu_list)  # 分子可以选择的元素

            zi=mu*n # 分子
            zi_list=list(str(zi)) # 2=<n<=79 , zi_list 起码是 4位

            if len(zi_list) >5: # 分子和 分母的位数 和大于10, 可以退出
                continue

            if len(zi_list)==4: # mu_list=[0,1,2,3,4] n=2 zi_list=[2,4,6,8] ,zi_list 第一位要补0  zi_list=[0,2,4,6,8]
                continue

            zi_set=set(zi_list)

            if len(zi_set) < len(zi_list): # zi_list 有重复元素, 不满足要求
                continue

            # 分母 和 分子 中所有数字都不相同
            if len( zi_set & mu_set )==0:

                res_list.append((zi_list,mu_list))


        return res_list

    def solve_deprecated(self, n):
        """
        对分子的所有情况 进行枚举 abcde , 通过 分子/n 得到分母, 判断 分母中的元素 是否是 分子中的元素的子集
         
        超时（TLE）
        
        :param n: 
        :return: 
        """

        # num_set=set(list(range(10)))

        num_set= set([ str(i) for i in range(10)] ) # 可选数字的集合 , 用字符表示

        res_list=[]

        for molecular_list in itertools.permutations(num_set, 5): # 0-9 的长度是 5 的全部的排列

            # molecular=IntDigit.ListToInt(molecular_list) # 分子

            molecular = int(''.join(molecular_list))

            denominator_num_set= num_set - set(molecular_list) # 分母 可以选择的元素

            if molecular % n==0: # 必须为 整数

                # print(molecular)
                denominator=molecular // n # 分母

                # denominator_list=IntDigit.IntToList(denominator)
                denominator_list= list(str(denominator))

                if len(denominator_list)==5:

                    if set(denominator_list)==denominator_num_set: # 集合中的元素匹配
                        res_list.append((molecular_list,denominator_list))

                elif len(denominator_list) == 4:  # 分母 的长度 为4, 说明 首位可以补0

                    denominator_list.append('0') # 补在 尾部也不影响判断

                    if set(denominator_list)==denominator_num_set:
                        res_list.append((molecular_list, denominator_list))

        return res_list

if __name__ == '__main__':

    sol = solutions()

    flag = 2  # TODO:提交代码记得修改 状态

    # IDE 测试 阶段：
    if flag == 1:

        print(sol.solve(62))

        # IDE 测试 阶段：
        # test = Test()

        # test.test_small_dataset(sol.solve)
        # test.test_large_dataset(sol.solve)


    # 提交 阶段：
    elif flag == 2:
        # 启动 windows power shell  提交
        # Get-Content inputs | python uva_725.py

        while True:

            n = int(input().strip())  # 输入数据 n

            if n==0:
                break

            res_list = sol.solve(n) # 分子 分母

            if len(res_list)==0:

                print('There are no solutions for {}.\n'.format(n))

            else:

                for res in res_list:

                    molecular= ''.join(res[0])
                    denominator=''.join(res[1])

                    print('{} / {} = {}'.format(molecular, denominator ,n ))

                print("\n")














