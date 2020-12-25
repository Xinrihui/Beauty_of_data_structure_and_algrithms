#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

import numpy as np


class Solution_Reference(object):

    def solve1(self, k , prices):
        """
        题解
        https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/solution/gu-piao-jiao-yi-xi-lie-cong-tan-xin-dao-dong-tai-g/
        
        还是 TLE 
        :param k: 
        :param prices: 
        :return: 
        """


        n = len(prices)
        if n <= 1: return 0

        if k >= n // 2:  # 退化为不限制交易次数
            profit = 0
            for i in range(1, n):
                if prices[i] > prices[i - 1]:
                    profit += prices[i] - prices[i - 1]
            return profit

        else:  # 限制交易次数为k
            dp = [[[None, None] for _ in range(k + 1)] for _ in range(n)]  # (n, k+1, 2)
            for i in range(n):
                dp[i][0][0] = 0
                dp[i][0][1] = -float('inf')
            for j in range(1, k + 1):
                dp[0][j][0] = 0
                dp[0][j][1] = -prices[0]
            for i in range(1, n):
                for j in range(1, k + 1):
                    dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i])
                    dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i])
            return dp[-1][-1][0]


    def solve2(self, k, prices):
        """
        题解
        https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/solution/si-chong-jie-fa-tu-jie-188mai-mai-gu-piao-de-zui-j/
        
        :param k: 
        :param prices: 
        :return: 
        """
        if not prices:
            return 0
        n = len(prices)
        # 当k非常大时转为无限次交易
        if k>n//2:
            dp0,dp1 = 0,-prices[0]
            for i in range(1,n):
                tmp = dp0
                dp0 = max(dp0,dp1+prices[i])
                dp1 = max(dp1,dp0-prices[i])
            return max(dp0,dp1)

        # 定义二维数组，交易了多少次、当前的买卖状态
        dp = [[0,0] for _ in range(k+1)]
        for i in range(k+1):
            dp[i][1] = -prices[0]
        for i in range(1,n):
            for j in range(k,0,-1):
                # 处理第k次买入
                dp[j-1][1] = max(dp[j-1][1],dp[j-1][0]-prices[i])
                # 处理第k次卖出
                dp[j][0] = max(dp[j][0],dp[j-1][1]+prices[i])
        return dp[-1][0]


class Solution:


    #   maxProfit(self, k: int, prices: List[int]) -> int:
    def solve(self, K,prices):
        """
        M2: 股票问题的 动态规划模板 
        
        时间复杂度 

        :param prices: 
        :return: 
        """

        N=len(prices)

        max_profit = 0

        if N <= 1:
            return 0

        # flag,profit=self.find_max_profit(K,prices)
        # if flag==True:
        #
        #     max_profit=profit
        #
        #     return max_profit

        prices = [0] + prices

        if K > N//2:
            for i in range(2,N+1):
                profit=prices[i]-prices[i-1]

                if profit>0:
                    max_profit+=profit
            return max_profit


        # dp=float('-inf')*np.ones((N+1,K+1,2),dtype=int) # 初始化为 -inf

        # dp=[[[float('-inf') for __ in range(2)] for __ in range(K+1)] for __ in range(N+1)] # time: 21s

        # dp=[[[-65536 for __ in range(2)] for __ in range(K+1)] for __ in range(N+1)] # TODO: 比 float('-inf') 快 但还是 TLE time: 21s

        dp = [[[-65536,-65536] for __ in range(K + 1)] for __ in range(N + 1)] # TODO: 少掉 1层循环 时间减少很多 time: 13s


        # 初始化  第1 天
        dp[1][0][0]=0 # hold
        dp[1][0][1] = -prices[1] # buy

        for i in range(2,N+1):

            for k in range( min(i//2,K)+1 ):

                # print('i:{},k:{}'.format(i,k))

                # 第i 天未持有 股票
                if k==0:
                    dp[i][k][0] = dp[i - 1][k][0]

                else:
                    dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k - 1][1] + prices[i])

                # print('dp[i][k][0]',dp[i][k][0])

                # 第i 天 持有 股票
                dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k][0] - prices[i])

                # print('dp[i][k][0]', dp[i][k][1])

        # print(dp)

        # max_profit=dp[N][K][0]

        for k in range(K+1): # TODO：时间不是消耗在这里
            max_profit=max(max_profit,dp[N][k][0])

        return max_profit


    def find_max_profit(self,K,prices):
        """
        当交易次数 K 很大时，
        每一天的利润(利润为正)我都可以 拿到, 即 返回 价差数组中 所有 >0元素 的和
        
        :return: 
        """

        # 1. 得到 价格差数组
        diff = [0] * len(prices) # 今日与昨日的 价格差

        for i in range(1, len(prices)):
            diff[i] = prices[i] - prices[i - 1]

        # print('nums:',nums)

        n = len(diff)

        nums=diff

        # 2. 找出 连续为正数 的 子数组

        start = []  # 子数组的开始位置
        length = []  # 子数组的 长度

        for i in range(1, n ):

            if nums[i] > 0 and nums[i - 1] <= 0:  # 第 1个 正数

                start.append(i)
                length.append(1)  # 子串的 长度为1

            if nums[i] > 0 and nums[i - 1] > 0:  # 连续为 正数

                length[-1] += 1  # 子串的 长度 +1

        # print('start:', start)
        # print('length:', length)

        # 3. 判断 k 是否足够大
        print('length of start:',len(start))

        if K<len(start):
            return False,0

        sums_sub = [0] * len(start)  # 计算 子数组的 和
        for j in range(len(start)):
            sums_sub[j] = sum(nums[start[j]:start[j]+length[j]])

        print('sums_sub:', sums_sub)

        return True,sum(sums_sub)


class Solution_Deprecated:

    #   maxProfit(self, k: int, prices: List[int]) -> int:
    def solve(self, k,prices):
        """
        M1: 转换为 求 k个子数组的和 最大问题
        
        时间复杂度 

        :param prices: 
        :return: 
        """
        if len(prices)<=1:
            return 0

        diff = [0] * len(prices)

        for i in range(1, len(prices)):
            diff[i] = prices[i] - prices[i - 1]

        print('diff:',diff)

        return max(0, self.max_sub_k_array(diff,k))  # 若 利润<0 则输出0

    def max_sub_k_array(self,nums,k):
        """
        k 个子数组 的最大和 
        
        0 <= k <= 10^9
        0 <= prices.length <= 10^4

        
        :param nums: 
        :return: 
        """
        if k==0:
            return 0

        # 1. 记录 所有子数组的和

        n=len(nums)

        nums=[float('-inf')]+nums

        dp=np.zeros((n+1,n+1),dtype=int)

        # dp=[[0 for __ in range(n+1)] for __ in range(n+1)]

        for l in range(1,n+1):
            for i in range(1,n-l+2):

                if l==1:
                    dp[l][i]=nums[i]
                else:
                    dp[l][i] = dp[l-1][i]+ nums[i+l-1]

        # print(dp)

        # 2. 找出 连续为正数 的 子数组

        start=[] # 子数组的开始位置
        length=[] # 子数组的 长度


        for i in range(1, n+1):

            if nums[i]>0 and  nums[i-1]<=0: # 第 1个 正数

                start.append(i)
                length.append(1) # 子串的 长度为1

            if nums[i]>0 and  nums[i-1]>0: # 连续为 正数

                length[-1]+=1  # 子串的 长度 +1

        print('start:',start)
        print('length:',length)


        sums=[0]*len(start)# 记录 子数组的 和
        for j in range(len(start)):
            sums[j]=dp[length[j]][start[j]]

        print('sums:',sums)

        # 3. 判断 k 是否足够大
        if k>= len(start):

            return sum(sums)


        #4. 递归找出  把 nums 分为k 段 的最优方案

        self.max_merge_sum=0

        s=0 # 分段 起点
        prev_sum = 0

        current_num=1 # 已经分好的 段的个数

        for e in range(s+1,len(start)+1): # e: 分段的终点

            self.__process(dp,start,length,k,s,e,prev_sum,current_num)

        #5.
        sums_sorted=sorted(sums,reverse=True)

        res=max(sum(sums_sorted[0:k]),self.max_merge_sum)

        return res



    def __process(self,dp,start,length,k,s,e,prev_sum,current_num):

        print('s:{},e:{},current_num:{},sub:{},prev_sum:{}'.format(s, e, current_num, start[s:e],prev_sum))

        end_idx= start[e-1]+length[e-1]
        start_idx=start[s]
        length_merge=end_idx-start_idx

        prev_sum+=dp[length_merge][start_idx]

        # 递归结束条件
        # if current_num == k:
        if current_num==k and e==len(start): # k+1

            print('over:',prev_sum)

            if prev_sum > self.max_merge_sum:

                self.max_merge_sum=prev_sum

                return

        s=e
        current_num+=1

        for e in range(s+1,len(start)+1): # e: 分段的终点

            self.__process(dp,start,length,k,s,e,prev_sum,current_num)



class Test:
    def test_small_dataset(self, func):

        assert func( 2,[3,2,6,5,0,3]) == 7

        assert func(2,[2,4,1]) == 2

        k = 1
        prices = [6, 1, 6, 4, 3, 0, 2]
        assert func(k, prices) == 5

        k = 2
        prices = [5, 2, 3, 0, 3, 5, 6, 8, 1, 5]
        assert func(k, prices) == 12

        k = 4
        prices = [5, 2, 3, 0, 3, 5, 6, 8, 1, 5]
        assert func(k, prices) == 13

        k = 6
        prices = [5, 2, 3, 0, 3, 5, 6, 8, 1, 5]
        assert func(k, prices) == 13

        k = 2
        prices = [6, 5, 4, 8, 6, 8, 7, 8, 9, 4, 5]
        assert func(k, prices) == 7

        k =2
        prices = [3, 5, 3, 4, 1, 4, 5, 0, 7, 8, 5, 6, 9, 4, 1]
        assert func(k, prices) == 13

        # TODO: 边界条件
        # assert func(None) == None

        assert func(1,[1]) == 0

    def read_test_case_fromFile(self,dir):
        """
        读取 本地目录的 测试数据 
        
        :param dir: 
        :return: 
        """

        with open(dir,'r',encoding='utf-8') as file:  # utf-8 解码

            K1=int(file.readline().strip())
            print('K1: ', K1)

            l1=file.readline().strip()[1:-1].split(',')

            l1= [int(ele) for ele in l1]

            print('l1:',l1)

            return K1,l1




    def test_large_dataset(self, func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题

        Limits
        0 <= K <= 10^9
        0 <= prices.length <= 10^4

        :param func: 
        :return: 
        """

        N = int( pow(10, 4))

        # K=int(pow(10, 9))
        K=1000

        max_v = 1000

        l = list(np.random.randint(max_v, size=N))

        # start = timeit.default_timer()
        # print('run large dataset: ')
        # func(K,l)
        # end = timeit.default_timer()
        # print('time: ', end - start, 's')

        dir = 'large_test_case/188_1'
        K, l1=self.read_test_case_fromFile(dir)

        start = timeit.default_timer()
        print('run large dataset:{} '.format(dir))
        func(K,l1)
        end = timeit.default_timer()
        print('time: ', end - start, 's')

        dir = 'large_test_case/188_2'
        K, l1 = self.read_test_case_fromFile(dir)

        start = timeit.default_timer()
        print('run large dataset:{} '.format(dir))
        func(K, l1)  # 12.047259273 s
        end = timeit.default_timer()
        print('time: ', end - start, 's')


if __name__ == '__main__':
    sol = Solution()

    sol_ref=Solution_Reference()

    # IDE 测试 阶段：

    # print(sol.max_sub_k_array([-3,2,-1,3,-3,2,4,-1,2],3))

    # print(sol.solve(2,[3,2,6,5,0,3]))
    # print(sol.solve(2,[2,4,1]))

    k=1
    prices=[6, 1, 6, 4, 3, 0, 2]
    # print(sol.solve(k,prices))

    k =4
    prices =[5, 2, 3, 0, 3, 5, 6, 8, 1, 5]
    # print(sol.solve(k, prices))

    k =2
    prices =[6, 5, 4, 8, 6, 8, 7, 8, 9, 4, 5]
    # print(sol.solve(k, prices))


    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    test.test_large_dataset(sol.solve)

    # test.test_large_dataset(sol_ref.solve1)

    # test.test_large_dataset(sol_ref.solve2)










