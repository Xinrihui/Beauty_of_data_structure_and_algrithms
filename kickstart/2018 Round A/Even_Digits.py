#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import *

class Test:

    def test_even_digits(self,func):

        assert  func(42)==0

        assert  func(11)==3

        assert  func(1)==1


        assert func(2018) == 2

        assert func(2234424) == 5536

        assert func(12422886) == 3533998

        assert func(246202) == 0

        assert func(923) == 35

        assert func(4249231) == 343

        assert func(4289231) == 343

        assert func(8889231) == 343

        assert func(86912) == 24

        assert func(6488962) == 74

        assert func(88892) == 4

        assert func(91112) == 2224


        return True

class solutions:
    """
    
    Even Digits
    
    给定一个 正整数 N  有一个 按钮 对这个正整数N 进行 +1 和 -1 ， 求 最少 按几次 按钮 能使得 这个  整数的 每一位 都是 偶数
    
    
    ref: https://codingcompetitions.withgoogle.com/kickstart/round/0000000000050edf/00000000000510ed
    """

    def check(self,n):
        """
         判断 一个 整数的 每一位 是否 都是 偶数 
        :param n: 321
        :return: 
        """
        while (n > 0):

            low_bit = (n % 10)  # 最低位的值

            if low_bit % 2 != 0:  # 有一位不是 偶数
                return False

            n = n // 10 # 整除

        return True

    def listToInt(self, nums):
        """
        将列表 转换为 一个 整数
        :param nums: [4,4,2,4]
        :return: 4424
        """
        nums=nums[::-1]

        res=0
        for i in range(len(nums)):

            res+=(nums[i] * pow(10,i))

        return res

    def even_digits_naive(self,N):
        """
        暴力 枚举，先解决 小数据集
        
        :param N: 
        :return: 
        """
        i=0
        while True:

           if self.check(N-i) or self.check(N+i): # 判断 +i 或者 -i 是否能满足 全部位是 偶数的要求
                break

           i+=1

        return i


    def even_digits(self, N):
        """
        贪心法 
        
        1.对 N 往下找 （不断 -1），找 最大的数 满足 所有位都是 偶数 的条件
        
        2234424  =>  2228888 |2234424 - 2228888|= 5536‬
        
        12422886 => 08888888 |12422886-08888888|= 3533998
        
        246202 => 246202  =0
        
        
        2.对 N 往上找（不断 +1） ，找 最小的数 满足 所有位都是 偶数 的条件
              
              找到的数为
        22 3 4424  =>  22 4 0000  |2240000 - 2234424|= 5576
        
        1 2422886 =>  2 0000000 |20000000 - 12422886|= 7577114
        
            2.1 奇数位 为 9 需要特殊处理：
                                                    N 往下找
            9 23 => 20 00  | 2000 - 923 | = 1077  & |923 - 888|=35‬
            
            424 9 231 => 426 0 000   | 4260000 - 4249231 | = 10769‬  & |4249231- 4248888|= 343
            
            428 9 231 => 440 0 000   | 4400000 - 4289231 | = 110769 & |4289231 - 4288888 | = 343
            
            888 9 231 => 2000 0 000  | 20000000 - 8889231 | = 11110769 & |8889231-8888888| = 343
            
            ------------
            86912 => 88000  |88000-86912|=1088 & |86912 - 86888|= 24
            
            6488962 => 6600000  | 6600000 - 6488962|= 111038 & |6488962 - 6488888|=74
        
            88892 => 200000   |200000-88892|= 111108 & |88892-88888|=4
             
            91112 => 200000  |200000-91112|=108888  & |91112-88888|=2224
        
        3. 对比 1. 2. 中找到的数，选择 与 N 差距小的 输出 
        
        :param N: 
        :return: 
        """

        N_list= list(map(int, list(str(N)))) # [2,0,1,8]

        first_even_digit=None
        first_even_digit_index=-1

        for index,digit in enumerate(N_list):

            if digit % 2 !=0: # 该位为 奇数

                first_even_digit=digit
                first_even_digit_index=index

                break
        else: # for 正常 退出，说明每一位都是 偶数

            return 0


        # 1. 对 N 往下找，找 最大的数 满足 所有位都是 偶数 的条件

        slice =N_list[first_even_digit_index+1:] # N_list=[2,2,3,4,4,2,4] slice=[4,4,2,4]

        # 22 3 4424  =>  22 2 8888
        max_num= self.listToInt(len(slice)*[8])

        look_up_value= self.listToInt([1]+slice)-max_num # 1 4424 - 8888 = 5536

        # 2.对 N 往上找（不断 +1） ，找 最小的数 满足 所有位都是 偶数 的条件

        # 22 3 4424  =>  22 4 0000

        if first_even_digit!=9 :

            min_num = self.listToInt([1]+len(slice) * [0]) # 1 0000

            look_down_value= min_num - self.listToInt(slice) # 1 0000 - 4424=5576

        else :

            # 奇数位 为 9 需要特殊处理：
            # 9 23 => 20 00
            # 424 9 231 => 426 0 000
            # 428 9 231 => 440 0 000
            # 888 9 231 => 2000 0 000

            if first_even_digit_index ==0: # 9 出现在 首位
                # 9 23 => 20 00
                look_down_value= self.listToInt([2]+len(N_list) * [0])- N # 20 00 - 9 23

            else:
            # eg.
            #index:012
            #      428 9 231 => 440 0 000

            # 从9的 前一位开始 找，找到第一个出现的8

                i=first_even_digit_index-1 # i=2

                while i>=0: # 若 一直没找到8 退出时 i=-1

                    if N_list[i]!=8: # 遇到 某位 不是8 的输出标号
                        break

                    i -= 1

                # i=1
                look_down_value = self.listToInt([2] + (len(N_list)-i-1) * [0])-self.listToInt(N_list[i+1:])  # 20 0 000- 8 9 231

        # print("N: %d ; look_up_value: %d ; look_down_value: %d" %(N,look_up_value,look_down_value)) #TODO: 提交记得 关掉输出

        return min(look_up_value,look_down_value)

if __name__ == '__main__':

    sol=solutions()


    # sol.even_digits(2234424)

    # IDE 测试 阶段：
    # test=Test()
    # test.test_even_digits(sol.even_digits)


    # 提交 阶段：
    T = int(input()) # 一共几个 测试数据
    for t in range(1, T + 1):

        N=int(input()) # 每一个 测试的 N

        res = sol.even_digits(N)
        print('Case #{}: {}'.format(t, res))








