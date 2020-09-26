#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import *

import timeit

class Test:

    def test_small_dataset(self,func):

        N, K, P=3,1,2
        rule_list=[(2,2,1)]

        assert  func(N, K, P,rule_list)=='011'

        N, K, P=3,1,3
        rule_list=[(2,2,1)]

        assert  func(N, K, P,rule_list)=='110'


        N, K, P=3,2,2
        rule_list=[(2,2,1),(1,1,1)]

        assert func(N, K, P, rule_list) == '111'


        N, K, P=4,2,3
        rule_list=[(1,1,1),(3,3,1)]

        assert func(N, K, P, rule_list) == '1110'


        N, K, P=100,2,int(1e6)
        rule_list=[(50,50,1),(51,51,1)]

        assert  func(N, K, P, rule_list)=='0000000000000000000000000000000000000000000000000110000000000000000000000000000011110100001000111111'
        # assert  func()==''


    def test_large_dataset(self,func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题
        
        Limits
        1 ≤ T ≤ 100.
        Time limit: 20 seconds per test set.
        Memory limit: 1 GB.
        
        1 ≤ N ≤ 100.
        1 ≤ K ≤ 100.
        1 ≤ P ≤ min(10^18, the number of bit strings that obey all of the constraints).
        
        1 ≤ Ai ≤ Bi ≤ N for all 1 ≤ i ≤ K.
        0 ≤ Ci ≤ N, for all 1 ≤ i ≤ K.
        (Ai, Bi) ≠ (Aj, Bj), for all 1 ≤ i < j ≤ K.
        
        :param func: 
        :return: 
        """

        N, K, P=100,2,int(1e16)
        rule_list=[(50,50,1),(51,51,1)]


        start = timeit.default_timer()
        print('run large dataset: ')

        print(func(N, K, P,rule_list))

        end = timeit.default_timer()
        print('time: ', end - start, 's')




class solutions:
    """
   
    """

    def _f(self, values, threshold):

        pass


    def solve_naive(self, N, K, P, rule_list):
        """
        非递归法 解 小数据集

        A=B  C=1(C=0) 相当于 对 A 位 置 1 或者 0
        
        1.遍历 rule_list, 按照 rule 中的 A, B 把相应的位 填写 在pattern中，同时记录哪些位被 填写过(change_flag)
        
        2. 此时，相当于 限定了若干格子里面的到底是 0还是 1了，剩下的格子 要按照 字典序 升序来填写
        
        eg.
        [1][][1][]  ( 模板 )
        字典序 升序 填写
        
        P=1            P=2          P=3      
        [1][0][1][0]  [1][0][1][1]  [1][1][1][0]
        
        把 P-1 以二进制 表示 填写 到剩余 空着的 格子中:
        
        P=3  bin(P-1)='0b10'  剩余 空着的两个格子 [1][0]
        
        res_list=[1,1,1,0]
        
        ref: 
        http://www.calvinneo.com/2017/11/14/Kickstart2018B/
        
        :param N: 
        :param K: 
        :param P: 
        :param rule_list: 
        :return: 
        """
        # rule_list=sorted(rule_list,key=lambda t: t[0])

        pattern=[0]*N
        change_flag=[0]*N # 被规则 更改后 即为 1

        for rule in rule_list:

            A,B,C=rule

            pattern[A-1]=C
            change_flag[A-1]=1

        # 把 P-1 以二进制 表示 填写 到剩余 空着的 格子中：

        P_binary_list=list(bin(P-1))[2:] # P=3 P-1=2 P_binary=['1','0']
        j=len(P_binary_list)-1

        res_list=pattern

        for i in range(N-1,-1,-1): # 从尾巴开始 向前填

            if j==-1: # P_binary_list 中的元素 已经 填完了
                break

            if change_flag[i]==0: # 不是模板中 定了的元素 才能往里填
                res_list[i]= int(P_binary_list[j])
                j-=1


        return ''.join(map(str,res_list))

    def solve_naive_Deprecated(self, N, K, P,rule_list):
        """
        递归法解 小数据集
        
        A=B  C=1(C=0) 相当于 对 A 位 置 1 或者 0
        
        1.rule_list 根据 A 排序 
        2.遍历 rule_list, 按照 rule 中的 A, B 把相应的位 填写 在pattern中，同时记录哪些位被 填写过(change_flag)
        
        3. 此时，相当于 限定了若干格子里面的到底是 0还是 1了，剩下的格子 要按照 字典序 升序来填写
        
        eg.
        [][1][]  ( 模板 )
        字典序 升序 填写
        
        P=1        P=2        P=3      P=4
        [0][1][0]  [0][1][1] [1][1][0] [1][1][1]
        
        
        4.利用递归的 方法 生成 所有的 字符串的组合：
        
        (1) 第 0 位 可取 0 和 1
            '0'  '1'
        (2) 第 1 位 只可取  1 
            '01' '11'
        (3) 第 2 位  可取 0 和 1
            '010' '011' '110' '111' 
        
        4. 返回  第 P个字符串 
        
        TODO : TTL 超时错误
         
        :param N: 
        :param K: 
        :param P: 
        :param rule_list: 
        :return: 
        """

        rule_list=sorted(rule_list,key=lambda t: t[0])

        self.N = N
        self.P=P

        self.pattern=[0]*N
        self.change_flag=[False]*N #被规则 更改后即为 True

        for rule in rule_list: # rule=(2,2,1)

            A,B,C=rule

            self.pattern[A-1]=C
            self.change_flag[A-1]=True

        # pattern=[0,1,0]
        # change_flag=[False,True,False]

        self.NO=0
        self.res_NO=''

        self._dfs_recursive(0,'') # 递归的方法


        return self.res_NO


    def _dfs_recursive(self,i,res):
        """
        1. 递归的边界：
        
            i=0 res=''
            i=1 res='0'
            i=2 res='01'
            i=3 res='010'
            
        2. 递归的 退出条件：
            (1) 字符串长度 达到 N:
                i == self.N
            
            (2) 按照 字典序 升序 已经 生成了 第 P 个 字符串:  
                self.NO==self.P
        
        问题：超时错误 TTL 
        
        :param i: 
        :param res: 
        :return: 
        """

        if self.NO==self.P:

            return

        if i < self.N:

            if self.change_flag[i]==False:
                self._dfs_recursive(i+1,res+'0')
                self._dfs_recursive(i + 1, res + '1')
            else:
                self._dfs_recursive(i + 1, res + str(self.pattern[i]))

        elif i == self.N:
            self.NO +=1 # TODO：超时错误(TTL) ; 当 P=10^18 此语句 要执行 10^18 次，凉凉了
            self.res_NO=res

            return



    def solve(self, N, K, P,rule_list):

        pass


if __name__ == '__main__':

    sol=solutions()


    # IDE 测试 阶段：
    test=Test()
    test.test_small_dataset(sol.solve_naive)
    #
    # test.test_large_dataset(sol.solve_naive)

    # 提交 阶段：
    # pycharm 开命令行 提交
    #E:\python package\python-project\Beauty_of_data_structure_and_algrithms\kickstart\2018 Round A>python Scrambled_Words.py < inputs
    # Case #1: 4

    # T = int(input().strip()) # 一共T个 测试数据
    #
    # for t in range(1, T + 1):
    #
    #     N, K, P = [int(i) for i in input().split(' ')]
    #
    #     rule_list=[]
    #     for i in range(K):
    #         A, B, C=[int(i) for i in input().split(' ')]
    #         rule_list.append((A,B,C))
    #
    #     res = sol.solve_naive(N, K, P,rule_list)
    #
    #     print('Case #{}: {}'.format(t, res))









