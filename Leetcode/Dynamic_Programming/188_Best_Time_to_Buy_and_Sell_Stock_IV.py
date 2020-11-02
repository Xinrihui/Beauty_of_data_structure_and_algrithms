#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

import numpy as np


class Solution:

    #   maxProfit(self, k: int, prices: List[int]) -> int:
    def solve(self, k,prices):
        """

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



        #4. 递归找出 最优地 把 nums 分为k 段

        # self.max_merge_sum=0
        #
        # s=0 # 分段 起点
        # prev_sum = 0
        #
        # current_num=1 # 已经分好的 段的个数
        #
        # for e in range(s+1,len(start)+1): # e: 分段的终点
        #
        #     self.__process(dp,start,length,k,s,e,prev_sum,current_num)

        #5.
        sums_sorted=sorted(sums,reverse=True)

        # res=max(sum(sums_sorted[0:k]),self.max_merge_sum)

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

        k = 2
        prices = [6, 5, 4, 8, 6, 8, 7, 8, 9, 4, 5]
        assert func(k, prices) == 7

        k =2
        prices = [3, 5, 3, 4, 1, 4, 5, 0, 7, 8, 5, 6, 9, 4, 1]
        assert func(k, prices) == 13

        # assert func([7, 6, 4, 3, 1]) == 0

        # assert func("cbbc") == 'cbbc'
        #
        # assert func("abcd") == 'a'

        # TODO: 边界条件
        # assert func(None) == None

        assert func(1,[1]) == 0

    def test_large_dataset(self, func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题

        Limits


        :param func: 
        :return: 
        """

        N = int(2 * pow(10, 4))
        max_v = int(pow(10, 9))

        l = np.random.randint(max_v, size=N)
        l1 = list(l)

        start = timeit.default_timer()
        print('run large dataset: ')
        func()
        end = timeit.default_timer()
        print('time: ', end - start, 's')


if __name__ == '__main__':
    sol = Solution()

    # IDE 测试 阶段：

    print(sol.max_sub_k_array([-3,2,-1,3,-3,2,4,-1,2],3))

    # print(sol.solve(2,[3,2,6,5,0,3]))
    # print(sol.solve(2,[2,4,1]))

    k=1
    prices=[6, 1, 6, 4, 3, 0, 2]
    # print(sol.solve(k,prices))

    k =2
    prices =[5, 2, 3, 0, 3, 5, 6, 8, 1, 5]
    # print(sol.solve(k, prices))

    k =2
    prices =[6, 5, 4, 8, 6, 8, 7, 8, 9, 4, 5]
    # print(sol.solve(k, prices))

    k=2
    prices =[3, 5, 3, 4, 1, 4, 5, 0, 7, 8, 5, 6, 9, 4, 1]

    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










