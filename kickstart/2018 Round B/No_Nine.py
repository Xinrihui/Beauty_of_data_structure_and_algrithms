#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import *

import timeit

class Test:

    def test_small_dataset(self,func_validate,func):

        res,_=func_validate(16,26)
        assert func(16,26)==res

        res, _ = func_validate(88,102)
        assert  func(88,102)==res

        res, _ = func_validate(1, 2)
        assert  func(1,2)==res

        res, _ = func_validate(1, 10)
        assert func(1, 10) == res

        res, _ = func_validate(10, 100)
        assert func(10, 100) == res

        # res, _ = func_validate(900, 1000)
        # assert func(900, 1000) == res


    def test_large_dataset(self,func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题
        
        Limits
        
        1 ≤ T ≤ 100.
        Time limit: 60 seconds per test set.
        
        Memory limit: 1 GB.
        
        F does not contain a 9 digit. 
        F is not divisible by 9.  
        
        
        L does not contain a 9 digit.
        L is not divisible by 9.
        
        边界条件 十分友好，F 和 L 都是 no-nine 数
        
        Small dataset (Test set 1 - Visible)
        1 ≤ F < L ≤ 10^6. --这个数量级 说明可以暴力求解 
        
        Large dataset (Test set 2 - Hidden)
        1 ≤ F < L ≤ 10^18. -- 这个 数量级 暴力 遍历 就凉凉了 
        
        :param func: 
        :return: 
        """

        F=1
        L=int(1e6)

        start = timeit.default_timer()
        print('run large dataset: ')
        func(F,L)
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
        return n%10

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

        digit_list=[]

        while n > 0:
            digit = n % 10

            digit_list.append(digit)

            n = n // 10

        return digit_list[::-1]

    @staticmethod
    def IntHexConversion(n,x):
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
        import  functools

        return  functools.reduce(lambda total, d: 10 * total + d, digit_list, 0)

    @staticmethod
    def getNthDigit(num,N):
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
    No Nine
    
    在 [F,L] 中 统计 满足 "No Nine" 数字 条件的 元素个数
    
    "No Nine" 数字 条件：
    1.每一位 都不是 9
    AND
    2. 不能被9整除 
    
    将问题转换为：
    1. 统计 [1,F] 中 满足 "Nine" 数字 条件的元素个数 F_Nine，
       则 满足 "No Nine" 数字 条件的 元素个数 为 F-F_Nine
       
    2. 统计 [1,L] 中 满足 "Nine" 数字 条件的元素个数 L_Nine，
       则 满足 "No Nine" 数字 条件的 元素个数 为 L-L_Nine
       
    3. 在 [F,L] 中 ，满足  "No Nine" 数字 条件的 元素个数 为 (L-L_Nine)-(F-F_Nine) 
    
    "Nine" 数字 条件：
    1. 任意 1位 为 9 
    OR
    2. 能被 9 整除 
    
    满足 "Nine" 数字 条件 可以转换为 ：A+B-C
     A = 任意 1位 为 9 
     B = 能被 9 整除
     C=  某一位 为 9 并且 能被 9 整除
     
    
    """

    def _count_divisible_by_nine (self,upper_bound):
        """
        [1,upper_bound] 中 可以被 9 整除的 元素的 个数  
        
        :param upper_bound: 100 
        :return: 11
        """

        return upper_bound//9

    def _count_length_digits_contain_nine(self,length):
        """
        位数在 [0,length] 中的 整数，满足任意 1 位包含 9 的 元素 个数
        
        eg. 
        
        length=0  满足条件的集合：[] count=0
        
        length=1  满足条件的集合：[9]  count=1 
        
        length=2  满足条件的集合：[09,19,29,39,...99]  count=19 
        
        eg1. 
        length=3
        <= 3 位数字 
        [][][] 选一位 填入9 ，剩下 两位 在 0-9 之间任取 ，排列数为 3*10*10;
        999 被重复计算了 2次, 因此，所有可能为 3*10*10-2
        
        
        :param length: 
        :return: 
        
        """
        count=0

        if length>=1:
            count= length*pow(10,length-1)-(length-1)

        return  count

    def _count_digit_contain_nine(self,upper_bound):
        """
        [1,upper_bound] 中  任意 1 位包含 9 的 元素的个数
        
        eg1.
        
        upper_bound= 849
        
        upper_bound_list[0]=8, 首位 取 0-7(高位 <8 整体元素必然 <849) 一共有 8种选择, 剩下 2位 任意 1位含有 9 即可, 
        元素个数为 8*(2*10-1)
        
        upper_bound_list[1]=4, 第 1 位 取 0-3 一共有 4种选择, 剩下 1位 必须为 9,  
        元素个数为 4*1
        
        upper_bound_list[2]=9, 第 2 位(最后一位) 取 0-8 一共有 9种选择, 但是没有后面的位 来填 9了
        元素个数为 9*0
        但是 因为  upper_bound_list[2] 本身就是 9 所以 
        元素个数为 0+1 
        
        count=8*(2*10-1) + 4*1 + 1
        
        eg2.
        upper_bound= 895
        
        upper_bound_list[0]=8, 首位 取 0-7 一共有 8种选择, 剩下 2位 任意 1位含有 9 即可, 
        元素个数为 8*(2*10-1)
        
        upper_bound_list[1]=9, 第 1 位 取 0-8 一共有 9种选择, 剩下 1位 必须为 9,  
        元素个数为 9*1
        
        由于 upper_bound_list[1] 本身就是 9, 不用再 迭代剩下的 位了, 
        即 89[] 只要满足 <=895 即可，所以最后 1 位可以填 0-5 
        元素个数为 9*1 + 6
        
        count= 8*(2*10-1)+ (9*1 + 6)
        
        :param upper_bound: 
        :return: count
        """

        upper_bound_digits=IntDigit.IntToList(upper_bound) # [8,4,9]

        length=len(upper_bound_digits) # 3

        count=0

        i=0
        while i < length:

            digit=upper_bound_digits[i]
            count += digit * self._count_length_digits_contain_nine(length-(i+1))

            if digit==9: # 剩下的 位 不用再迭代了

                tail_digits_int= IntDigit.ListToInt(upper_bound_digits[i+1:]) + 1 # 9 -> 0-9 一共10个

                count += tail_digits_int

                break

            i+=1

        return count

    def _count_digit_contain_nine_and_divisible_by_nine (self, upper_bound):
        """
        [1,upper_bound] 中  满足 "任意 1 位包含 9" 并且 "可以被 9 整除" 的元素的个数
        
        整数规则
        
        1.一个整数 的各个位的数字和 能被9整除，则这个数能被9整除。
        
        99: 9 + 9 = 18
        198: 1 + 9 + 8 = 18,
        171: 1 + 7 + 1 = 9
        3411:  3 + 4 + 1+ 1 = 9
        
        ref:https://www.cnblogs.com/youxin/p/3219622.html
        
        eg. 
        upper_bound=100 满足条件的集合 [9,90,99]  count=3 
        
        eg1. 
        length=1 
        <= 1位数字
        sum_bits 可能取值: [9] 
        满足条件 元素集合: res=[9]
        
        length =2 
        <= 2位数字
        sum_bits 可能取值: [9,18]
        sum_bits=9: res=[09,90]
        sum_bits=18: res=[99]
        
        length =3 
        <= 3位数字
        sum_bits 可能取值: [9,18,27]
        
        sum_bits=9: res=[009,090,900]
        
        sum_bits=18: 
        [][][] 任意一位 填9,  剩下的两位 填写的和为 9
          
        对于 后两位 的 解的个数 利用插值法： 
        111 0 111111 往 10个 空位中 插入 9 个 '1', 组合数为 C(9+1,9) =10 
        
        res_length= C(3,1) x C(9+1,9) = 30
    
        sum_bits=27: res=[999]
        
        :param upper_bound: 
        :return: 
        """

        count=0
        # TODO

        return count


    def no_nine_naive_validate(self, F,L):

        count=0

        no_nine_list=[]

        for i in range(F,L+1):

            if i % 9!=0 and '9' not in str(i):

                no_nine_list.append(i)
                count+=1

        return count,no_nine_list

    def no_nine_naive(self, F,L):

        count=0

        for i in range(F,L+1):

            if i % 9!=0 and '9' not in str(i): #TODO： 用字符串的 方法 遍历一个整数的 各个位 更快

                count+=1

        return count


    def no_nine(self, F,L):

        F= F-1

        # 在[1,F]中 满足 "任意 1 位包含 9"  或者  "可以被 9 整除" 的元素 个数
        F_nine= self._count_divisible_by_nine(F)+self._count_digit_contain_nine(F)-self._count_digit_contain_nine_and_divisible_by_nine(F)

        F_no_nine=F-F_nine

        # 在[1,L]中 满足 "任意 1 位包含 9"  或者  "可以被 9 整除" 的元素 个数
        L_nine=self._count_divisible_by_nine(L)+self._count_digit_contain_nine(L)-self._count_digit_contain_nine_and_divisible_by_nine(L)

        L_no_nine=L-L_nine

        return L_no_nine - F_no_nine


class solutions2:
    """
    No Nine
    
    在 [F,L] 中 统计 满足 "No Nine" 数字 条件的 元素个数
    
    "No Nine" 数字 条件：
    1.每一位 都不是 9
    AND
    2. 不能被9整除 
    
    将问题转换为：
    1. 统计 [1,F] 中 满足 "No Nine" 数字 条件的 元素个数 为 F_no_nine
    2. 统计 [1,L] 中 满足 "No Nine" 数字 条件的 元素个数 为 L_no_nine
    
    3. 题目的 约束条件 F 满足 "No Nine" 数字 所以 [F,L] 中 的 "No Nine" 数字 个数为 L_no_nine-F_no_nine + 1
        
    
    """

    def solve(self,F,L):

        F_no_nine=self.count(F)
        L_no_nine=self.count(L)

        return L_no_nine-F_no_nine +1

    def count(self,upper_bound):
        """
        
        [1,upper_bound] 中 满足 "No Nine" 数字 条件 的元素的个数
         
        eg.
        upper_bound=2088
        
        upper_bound_list[0]=1, 首位(第 0位) 取 0-1 (高位取 <1 ,整体元素必然 <1088 ) 一共有 2种选择, 
        剩下的 3位：
        第 1位 在 0-8 中取(显然不能取 9),一共有 9种选择 ;
        第 2位 在 0-8 中取(显然不能取 9);
        末位(第 3位 ) 在 0-8 中取 但是 肯定 有某一位 会导致 整体元素 被 9 整除，所以 一共有 (9-1)=8 种选择
        元素个数为 : 2*9*9*8
        
        upper_bound_list[1]=0, 第 1 位 取 null 一共有 0种选择,
        元素个数为 : 0
        
        upper_bound_list[2]=8, 第 2 位  取 0-7 一共有 8种选择, 
        末位(第 3位 ) 在 0-8 中取 但是 肯定 有某一位 会导致 整体元素 被 9 整除，所以 一共有 (9-1)=8 种选择
        元素个数为: 8*8 
        
        upper_bound_list[3]=8, 第 3 位 为末位, 枚举 末位 的所有可能：[2080,2081,..,2088], 判断整体元素是否能被 9 整除
        取 末位的数字:
        upper_bound%10=8 或者 upper_bound_list[3]
        
        upper_bound - upper_bound_list[3] = 2088 -8 =2080
         
        :param N: 
        :return: 
        """

        count=0

        L=len(str(upper_bound))

        for i,digit in enumerate(str(upper_bound)): # [1,0,8,8]

            if i < L-1: # i= 0,1,2

                count+= int(digit)*(9**(L-i-1-1))*8 # 根据题目的限制条件， 每一个 digit 不可能是9

            else: # 最后 一位 i=3

                for i in range( upper_bound - int(digit) ,upper_bound+1): # 枚举个位 的所有可能：[1080,1081,..,1088]

                    if i%9 !=0: # 不能被9 整除
                        count+=1

        return count



if __name__ == '__main__':

    sol=solutions()

    # print(IntDigit.IntToList(849))
    # print(IntDigit.ListToInt([8,4,9]))

    # print(sol._count_divisible_by_nine(100)) # 11

    # print(sol._count_digit_contain_nine(100)) # 19
    # print(sol._count_digit_contain_nine(109)) # 20
    # print(sol._count_digit_contain_nine(849)) # 157
    #
    # print(sol._count_digit_contain_nine(940)) # 211
    # print(sol._count_digit_contain_nine(890)) # 162


    # print(sol._count_digit_contain_nine_and_divisible_by_nine(849))
    # print(sol._count_digit_contain_nine_and_divisible_by_nine(900))
    # print(sol._count_digit_contain_nine_and_divisible_by_nine(906))
    # print(sol._count_digit_contain_nine_and_divisible_by_nine(909))

    # print(sol._count_digit_contain_nine_and_divisible_by_nine(990))
    # print(sol._count_digit_contain_nine_and_divisible_by_nine(996))
    # print(sol._count_digit_contain_nine_and_divisible_by_nine(999))
    # print(sol._count_digit_contain_nine_and_divisible_by_nine(1000))


    # print(sol._count_divisible_by_nine(88)) # 9
    # print(sol._count_digit_contain_nine(88)) # 8
    # print(sol._count_digit_contain_nine_and_divisible_by_nine(88)) # 1


    # print(sol.no_nine(16, 26))
    # print(sol.no_nine(88,102))

    # print(sol.no_nine_naive(16, 26))
    # print(sol.no_nine_naive(88, 102))

    sol2 = solutions2()

    # IDE 测试 阶段：
    test=Test()
    # test.test_small_dataset(sol.no_nine_naive_validate,sol2.solve)
    # test.test_large_dataset(sol.no_nine_naive)

    # 提交 阶段：
    # pycharm 开命令行 提交
    #E:\python package\python-project\Beauty_of_data_structure_and_algrithms\kickstart\2018 Round A>python Scrambled_Words.py < inputs
    # Case #1:

    T = int(input()) # 一共T个 测试数据

    for t in range(1, T + 1):

        F_L = [int(i) for i in input().split(' ')]

        F,L=F_L[0],F_L[1]

        res = sol2.solve(F,L)

        print('Case #{}: {}'.format(t, res))









