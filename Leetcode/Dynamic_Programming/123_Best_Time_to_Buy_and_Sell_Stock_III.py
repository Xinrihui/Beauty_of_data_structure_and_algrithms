#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit


import numpy as np

class Solution:

    #  maxProfit(self, prices: List[int]) -> int:
    def solve(self, prices):
        """

        时间复杂度 

        :param prices: 
        :return: 
        """

        diff=[0]*len(prices)

        for i in range(1,len(prices)):

            diff[i]=prices[i]-prices[i-1]

        # print('diff:',diff)

        return max(0,self.max_sub_two_array(diff)) # 若 利润<0 则输出0

    def max_sub_array(self,nums):
        """
        最大 连续子数组
        
        :param nums: 
        :return: 
        """

        L_nums=len(nums)

        nums=[0]+nums # 数组 前面加上0 ,生成新的数组, 不改变原来的 nums

        dp=[0]*(L_nums+1)

        for i in range(1,L_nums+1):

            if dp[i-1]>0:
                dp[i]=dp[i-1]+nums[i]

            else:
                dp[i] = nums[i]

        return dp

    def max_sub_two_array(self,nums):
        """
        
        :param nums: 
        :return: 
        """
        L_num=len(nums)

        dp_one=self.max_sub_array(nums)

        nums_rev=nums[::-1]
        dp_one_rev=self.max_sub_array(nums_rev)

        # print('dp_one:',dp_one)
        # print('dp_one_rev:',dp_one_rev)

        max_dp_one_rev={} #记录 dp_one_rev[1:L_num+1] 的最大元素

        max_ele=float('-inf')
        for j1 in range(1,L_num+1):

            if dp_one_rev[j1]>max_ele:
                max_ele=dp_one_rev[j1]

            max_dp_one_rev[j1] = max_ele

        # print('max_dp_one_rev:',max_dp_one_rev)

        max_sub_dp_one=float('-inf')
        max_sum=float('-inf')

        dp_one[0]=float('-inf')

        for i in range(0,L_num):

            j1=L_num-i

            max_sub_dp_one=max(max_sub_dp_one,dp_one[i])

            max_sum = max(max_sub_dp_one + max_dp_one_rev[j1],max_sum)

            # print(max_sub_dp_one + max_dp_one_rev[j1],max_sum)

        return max_sum




class Test:
    def test_small_dataset(self, func):

        assert func([3,3,5,0,0,3,1,4]) == 6

        assert func([1,2,3,4,5]) == 4

        assert func([7,6,4,3,1]) == 0

        # assert func("cbbc") == 'cbbc'
        #
        # assert func("abcd") == 'a'

        # TODO: 边界条件
        # assert func(None) == None

        assert func([1]) == 0

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

    # print(sol.max_sub_two_array([-3,2,-1,3,-3,2,4,-1]))

    # print(sol.solve([3,3,5,0,0,3,1,4]))

    # print(sol.solve([7,6,4,3,1]))
    #
    # print(sol.solve([1]))

    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










