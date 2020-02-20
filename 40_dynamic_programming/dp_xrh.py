#!/usr/bin/python
# -*- coding: UTF-8 -*-


from numpy import *

class solutions:

    def yh_triangle_dp(self,num):
        """
        对杨辉三角 进行改造，每个位置的数字可以随意填写，经过某个数字只能到达下面一层相邻的两个数字。
        假设你站在第一层（最高层），往下移动，我们把移动到最底层所经过的所有数字之和，定义为路径的长度。
        
        求从最高层移动到最底层的最短路径长度。
        by 动态规划 
        :param num: 
        :return: 
        """

        states=[num[0][0]]

        for i in range(1,len(num)):
            next_states=zeros(i+1, dtype=int)

            #第一个 和最后一个元素特殊处理
            next_states[0]=states[0]+num[i][0]
            next_states[-1] = states[-1] + num[i][-1]

            # 中间的元素 使用 状态转移方程
            for j in range(1,i):
                next_states[j]=min(states[j-1]+num[i][j],states[j]+num[i][j])

            states=next_states

        print(states)

        return min(states)

    def coins_select(self, coin_values, pay):
        """    
        假设我们有几种不同币值的硬币 v1，v2，……，vn（单位是元）。
        如果我们要支付 w 元，求最少需要多少个硬币。
        比如，我们有 3 种不同的硬币，1 元、3 元、5 元，我们要支付 9 元，最少需要 3 个硬币（3 个 3 元的硬币）。
        :param coin_values: [1,3,5]
        :param pay: 9
        :return: 3,[1,3,5]
        """
        col_num=pay+1

        states_num = zeros(col_num, dtype=int) # 记录达到 某价格 所需要的最少硬币的数量
        states_coin=zeros(col_num, dtype=int) # 记录达到 某价格 最后一次取的硬币的面值

        for i in  range(1,col_num):
            min_num=float('inf')

            num = float('inf')
            coin=None

            for value in coin_values:

                if i-value>=0:
                    num=states_num[i-value]+1

                if num<min_num:
                    min_num=num
                    coin=value

            states_num[i]=min_num
            states_coin[i]=coin

        # print(states_num)
        # print(states_coin)

        pay_num=states_num[-1] # 达到 最终状态 pay=9 所需要的 硬币的个数
        last_coin=states_coin[-1] # 达到 最终状态 最后一次取的硬币的 面值
        coin_list=[last_coin] # 把 最后一枚 加入到 硬币列表
        sum_value=pay

        for i in range(pay_num-1):
            sum_value=sum_value-last_coin
            last_coin= states_coin[sum_value]

            coin_list.append(last_coin)

        return pay_num,coin_list

class  solutions_edit_distance:


    def run_Levenshtein_distance_dp(self,s,t):
        """
        动态规划 计算两个字符串的编辑距离（莱文斯坦距离）
        ref: https://time.geekbang.org/column/article/75794
        :param s: 源字符串  'mitcmu'
        :param t: 目标字符串 'mtacnu'
        :return:  3
        """
        row_num=len(s)+1
        col_num=len(t)+1

        states = zeros((row_num,col_num), dtype=int)

        states[0,:]=arange(col_num)
        states[:,0]=arange(row_num)

        # print(states)

        for i in range(1,row_num):
            for j in range(1,col_num):
                if s[i-1]==t[j-1]:
                    states[i][j]=states[i-1][j-1]
                else:
                    states[i][j]=min(states[i-1][j-1]+1,states[i][j-1]+1,states[i-1][j]+1 )

        print(states)
        return states[-1][-1]

    def run_Longest_common_substring_length(self,s,t):
        """
        最长公共子串长度

        状态转移方程：
        如果：a[i]==b[j]，那么：max_lcs(i, j)就等于：
        max(max_lcs(i-1,j-1)+1, max_lcs(i-1, j), max_lcs(i, j-1))；
        如果：a[i]!=b[j]，那么：max_lcs(i, j)就等于：
        max(max_lcs(i-1,j-1), max_lcs(i-1, j), max_lcs(i, j-1))；
        
        ref: https://time.geekbang.org/column/article/75794
        
        :param s: mitcmu
        :param t: mtacnu
        :return:  4
        """

        row_num=len(s)
        col_num=len(t)

        states = zeros((row_num,col_num), dtype=int)


        for j in range(col_num):
            if s[0]==t[j]:
                states[0][j]=1
            elif j!=0:
                states[0][j]=states[0][j-1]
            else:
                states[0][j]=0

        for i in range(row_num):
            if s[i] == t[0]:
                states[i][0] = 1
            elif i != 0:
                states[i][0] = states[i-1][0]
            else:
                states[i][0] = 0

        # print(states)

        for i in range(1,row_num):
            for j in range(1,col_num):
                if s[i]==t[j]:
                    states[i][j]= max(states[i-1][j-1]+1,states[i][j-1],states[i-1][j])
                else:
                    states[i][j]=max(states[i][j-1],states[i-1][j],states[i-1][j-1] )

        print(states)

        return states[-1][-1]

    def run_Longest_common_substring_length_v2(self,s,t):
        """
        最长公共子串长度 (LCS)
        eg1. str=acbcbcef，str2=abcbced，则str和str2的最长公共子串为 abcbce，最长公共子串长度为6。
        
        状态转移方程：见《算法导论》
        空间复杂度为: O( length(s)*length(t)) 
        :param s: 
        :param t: 
        :return: 
        """

        col_num = len(t) + 1
        row_num=len(s)+1


        states_length = zeros((row_num,col_num), dtype=int)# 通过 states_length  记录 s 和 t 的前缀的 LCS 的长度


        for i in range(1,row_num):
            for j in range(1,col_num):
                if s[i-1]==t[j-1]: # 字符相同，则 LCS 长度 +1
                    states_length[i][j]= states_length[i-1][j-1]+1
                else:
                    states_length[i][j]=max(states_length[i][j-1],states_length[i-1][j] ) # 在左边或者上面挑一个大的

        # print(states_length)

        # 根据states_length 中 记录的 前缀 LCS 的长度，反推出LCS
        LCS=[]
        i=row_num-1
        j=col_num-1

        while i!=0 and j!=0:

            if s[i-1]==t[j-1]: #如果 字符相同 则在LCS中添加该字符
                LCS.append(s[i-1])
                i=i-1
                j=j-1
            else:
                if states_length[i][j-1] > states_length[i-1][j]:
                    j=j-1
                else:
                    i=i-1

        return states_length[-1][-1],LCS[::-1]

    def run_Longest_common_substring_length_v3(self,s,t):
        """
        降低空间复杂度为 O(length(t))
        :param s: 
        :param t: 
        :return: 
        """
        col_num = len(t) + 1
        row_num = len(s) + 1

        states_length = zeros( col_num, dtype=int)

        for i in range(1, row_num):

            next_states_length = zeros( col_num, dtype=int)

            for j in range(1, col_num):

                if s[i - 1] == t[j - 1]:  # 字符相同，则 LCS 长度 +1
                    next_states_length[j] = states_length[j - 1] + 1
                else:
                    next_states_length[j] = max(next_states_length[j - 1], states_length[j])  # 在左边或者上面挑一个大的

            states_length=next_states_length

        print(states_length)

        return states_length[-1]

    def run_Longest_increasing_substring_M1(self,s):
        """
        有一个数字序列包含 n 个不同的数字，如何求出这个序列中的最长递增子序列长度？
        eg.  2, 9, 3, 6, 5, 1, 7 这样一组数字序列，它的最长递增子序列就是 2, 3, 6, 7，所以最长递增子序列的长度是 4
        
        M1 : 可以先将数组排序，求两个数组的最大公共子序列，求得的结果即为最长递增子序列
        :param s: 
        :return: 
        """
        t=sorted(s)
        print(t)
        return self.run_Longest_common_substring_length_v2(s,t)

    def run_Longest_increasing_substring_M2(self, s):
        """
        有一个数字序列包含 n 个不同的数字，如何求出这个序列中的最长递增子序列长度？
        eg.  2, 9, 3, 6, 5, 1, 7 这样一组数字序列，它的最长递增子序列就是 2, 3, 6, 7，所以最长递增子序列的长度是 4

        M2 :
        states_length(i)表示 以 数值 s(i) 为结尾的最长递增子序列的长度
        
        显然 s(i) 为以s(i)为结尾的 递增子序列中的 最大的数值，
        
        对于每一个 states_length(i) ：
        遍历在 s(i)之前 的所有数值，找出最大的数值 s(j) 和 最小的数值 s(k)，
        case1 s(i) > s(j) ： states_length(i)= states_length(j)+1 
        case2 s(i) < s(k)  ： states_length(i)=1
        case3  s(i) 在 [s(k) , s(j)] 区间中：找满足 大于s(i) 中最大的数值 s(j_1)
                states_length(i)= states_length(j_1)+1 
        
        :param s: 
        :return: 
        """
        L=len(s)
        s=array(s)
        states_length = zeros(L, dtype=int)

        states_length[0]=1

        for i in range(1,L):

            current=s[i]

            prev_max=float('-inf')
            prev_max_id=None

            prev_min=float('inf')
            prev_min_id=None

            prev_smaller_max = float('-inf')  # 从 s[0:i] 中 过滤出比 current 小的元素，并在其中找最大的元素 和 其Index
            prev_smaller_max_id = None

            for j in range(i-1,-1,-1):

                if s[j] >= prev_max:
                    prev_max=s[j]
                    prev_max_id=j
                elif s[j] <prev_min:
                    prev_min = s[j]
                    prev_min_id = j

                if s[j] < current and s[j] > prev_smaller_max:
                    prev_smaller_max = s[j]
                    prev_smaller_max_id = j

            if current > prev_max:
                states_length[i] = states_length[prev_max_id] + 1

            elif current<prev_min:
                states_length[i] =  1

            else:
                states_length[i] = states_length[prev_smaller_max_id] + 1


            print(states_length)

        return states_length[-1]

    def run_Longest_increasing_substring_M2_v2(self, s):
        """
        有一个数字序列包含 n 个不同的数字，如何求出这个序列中的最长递增子序列长度？
        eg.  2, 9, 3, 6, 5, 1, 7 这样一组数字序列，它的最长递增子序列就是 2, 3, 6, 7，所以最长递增子序列的长度是 4

        M2 简化 :
        states_length(i)表示 以 数值 s(i) 为结尾的最长递增子序列的长度

        显然 s(i) 为以s(i)为结尾的 递增子序列中的 最大的数值，

        对于每一个 states_length(i) ：
        遍历在 s(i)之前 的所有数值，找出 最小的数值 s(k)，
        case1 s(i) >  s(k) ：找满足 大于s(i) 中最大的数值 s(j)
                states_length(i)= states_length(j)+1 
        case2 s(i) <= s(k) ： states_length(i)=1

        :param s: 
        :return: 
        """
        L = len(s)
        s = array(s)
        states_length = zeros(L, dtype=int)

        states_length[0] = 1

        for i in range(1, L):

            current = s[i]

            prev_min = float('inf')
            prev_min_id = None

            prev_smaller_max = float('-inf')  # 从 s[0:i] 中 过滤出比 current 小的元素，并在其中找最大的元素 和 其Index
            prev_smaller_max_id = None

            for j in range(i - 1, -1, -1):

                if s[j] < prev_min:
                    prev_min = s[j]
                    prev_min_id = j

                if s[j] < current and s[j] > prev_smaller_max:
                    prev_smaller_max = s[j]
                    prev_smaller_max_id = j


            if current <= prev_min:
                states_length[i] = 1

            else:
                states_length[i] = states_length[prev_smaller_max_id] + 1

            print(states_length)

        return states_length[-1]


if __name__ == '__main__':

    sol=solutions()
    coin_values=[1,3,5]
    # print(sol.coins_select(coin_values,9))
    # print(sol.coins_select(coin_values, 10))
    # print(sol.coins_select(coin_values, 7))

    nums = [[3], [2, 6], [5, 4, 2], [6, 0, 3, 2]]
    print(sol.yh_triangle_dp(nums))

    sol=solutions_edit_distance()
    s = "mitcmu"
    t = "mtacnu"

    # print(sol.run_Levenshtein_distance_dp(s,t))
    # print(sol.run_Levenshtein_distance_dp("kitten", "sitting"))
    # print(sol.run_Levenshtein_distance_dp("flaw", "lawn"))

    # print(sol.run_Longest_common_substring_length("mitcmu", "mtacnu"))

    # print(sol.run_Longest_common_substring_length_v2("abc", "ac"))
    # print(sol.run_Longest_common_substring_length_v2("kitten", "sitting"))
    # print(sol.run_Longest_common_substring_length_v2("flaw", "lawn"))
    # print(sol.run_Longest_common_substring_length_v2("acbcbcef", "abcbced"))

    # print(sol.run_Longest_common_substring_length_v2("mitcmu", "mtacnu"))
    # print(sol.run_Longest_common_substring_length_v3("mitcmu", "mtacnu"))

    # print(sol.run_Longest_increasing_substring_M2_v2([ 1, 5, 10, 2, 3, 4 ]))
    # print(sol.run_Longest_increasing_substring_M2_v2([2, 9, 3, 6, 5, 1, 7]))


