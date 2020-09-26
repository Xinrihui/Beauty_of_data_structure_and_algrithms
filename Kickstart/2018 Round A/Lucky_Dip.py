
#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import *

import timeit

class Test:

    def test_small_dataset(self,func):

        assert  func(4,0,[1,2,3,4])=='2.500000'

        assert  func(3,1,[1,10,1])=='6.000000'

        assert  func(3,5,[80000,80000,80000])=='80000.000000'


        assert func(1,1,[10]) == '10.000000'

        assert func(5, 3, [16, 11, 7, 4, 1]) == '12.358400'

    def test_large_dataset(self,func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题
        
        Limits
        
        Memory limit: 1GB.
        1 ≤ T ≤ 100.
        1 ≤ Vi ≤ 10^9.
        1 ≤ N ≤ 2 * 10^4.
        
        Small dataset (Test set 1 - Visible)
        Time limit: 20 seconds.
        0 ≤ K ≤ 1.
        
        Large dataset (Test set 2 - Hidden)
        Time limit: 60 seconds.
        0 ≤ K ≤ 5 * 10^4.
        
        :param func: 
        :return: 
        """

        N= int(2*pow(10,4))

        max_v= int(pow(10,9))

        K=int(5*pow(10,4))

        l = random.randint(max_v, size=N)
        l1 = list(l)

        start = timeit.default_timer()
        print('run large dataset: ')
        func(N,K,l1)
        end = timeit.default_timer()
        print('time: ', end - start, 's')



class solutions:
    """
    lucky dip
    
    可重复 的抽奖：每一次 抽到的 物品觉得不满意 可以放回 奖池中，并抽下一次；
    
    一共有 N 个 物品，价值为 value_list， 一共可以抽 K+1 次，求 在使用 最佳策略下 获得的 物品的价值 的期望
    
    ref:https://codingcompetitions.withgoogle.com/kickstart/round/0000000000050edf/0000000000050e1d
    
    eg1.
    N=4 K=0 values=[1,2,3,4]
    K+1=1 只能抽取 1 次
    
    期望 E= (1+2+3+4)/5 = 2.500000
    
    eg2.
    N=3 K=1 values=[1,10,1]
    
    期望 E= (5/9 * 10) + (4/9 * 1) = 6.000000
    
    eg3.
    N=3 K=5 values=[80000,80000,80000]
    
    期望 E=80000.000000
    
    eg4.
    N=1 K=1 values=[10]
    
    期望 E = 10.000000
    
    eg5.
    N=5 K=3 values=[16,11,7,4,1]
    
    期望 E = 12.358400
    
    """

    def _find_insert_idx(self, values, threshold):
        """
        二分查找，找到有序数组的插入位置，返回插入位置的 左端点

        :param values:有序数组  eg. values=[1,4,7,11,16]
        :param threshold:  阈值  eg. threshold=10
        :return:  index=3  10 应该插入 values 中 index=3 的位置 
        
        注意 边界情况：
        values=[1,4,7,11,16] threshold=17  index=5 
    
        values=[1,4,7,11,16] threshold=0  index=0 
        
        values=[8,8,8]  threshold=8  index=3 
         
        """
        # from bisect import bisect_left
        # return  bisect_left(values, threshold)

        start = 0
        end = len(values) - 1

        while start <= end:
            mid = (start + end) // 2
            if threshold < values[mid]:
                end = mid - 1
            else:
                if mid + 1 == len(values) or threshold < values[mid+1]:
                    return mid + 1
                else:
                    start = mid + 1
        return 0

    def lucky_dip_deprecated (self,N,K,values):
        """
        期望的 计算 出错
        
        :param N: 5
        :param K: 3
        :param values: [16,11,7,4,1]
        :return: 
        """

        # E=[0]*(K+1) # TODO:节约内存

        E= sum(values)/N #7.8

        values=sorted(values) # [1,4,7,11,16]

        prefix_sum_cache=[ sum(values[:i+1]) for i in range(len(values))] # 预先 求好 前缀的和， sum_cache=[1,5,12,23,39]

        for k in range(1,K+1): #k=1

            index=self._find_insert_idx(values,E) # index=3

            if index==len(values) or index== 0:
                # values=[80000,80000,80000] index=3

                break

            smaller_set=values[:index] #[1,4,7]
            larger_set=values[index:] # [11,16]

            # p= len(smaller_set) / N
            # p_k=pow( len(smaller_set)/N,k )
            # P_redip_1=  p_k *( 1/N )  # 若取到 则丢弃的 元素 ；P(V=1)=P(V=4)=P(V=7)
            # P_keep_1 = (1 - p_k * p) / len(larger_set)

            P_redip = pow( len(smaller_set)/N,k ) * (1 / N) # 若取到 则丢弃的 元素 ；P(V=1)=P(V=4)=P(V=7)

            P_keep = (1 - pow( len(smaller_set)/N,k+1 )) / len(larger_set)

            sum_smaller_set=prefix_sum_cache[index-1] #

            sum_larger_set= prefix_sum_cache[-1]-sum_smaller_set

            E= P_redip * sum_smaller_set + P_keep * sum_larger_set # TODO: 预先 求好前缀和 并缓存，可以 降低时间


        res = E

        # print(res) #TODO: 提交 时 注释

        return '{:.6f}'.format(res) # {:.6f} 保留 6位小数

    def lucky_dip_naive(self, N, K, values):
        """
        不计算概率，直接计算期望 

        :param N: 5
        :param K: 3
        :param values: [16,11,7,4,1]
        :return: 12.3584
        """

        E = sum(values) / N  # 7.8


        for k in range(1, K + 1):  #

            #sum_val=0 # 大数据级，报溢出： RuntimeWarning: overflow encountered in long scalars
            sum_val= float64(0)

            for val in values: # TODO： 大数据集 报 TTL（超时），尝试去除 内层循环

                if val >=E:
                    sum_val+=val

                else:

                    sum_val+=E

            E=sum_val/len(values)

        res = E

        # print(res) #TODO: 提交 时 注释

        return '{:.6f}'.format(res)  # {:.6f} 保留 6位小数


    def lucky_dip(self, N, K, values):
        """
        不计算概率，直接计算期望 
        
        解决 lucky_dip_naive 在 大数据集下的 TTL 问题：
        
        1.消除对 values 的遍历： 
            
            step1. 对 values 快速排序 ， 时间复杂度 O(nlogn)
            
            step2. 对 排序后的 values 预先计算 后缀和
            
            step3. 排序后的 values 二分查找 期望 E 应该在的位置 index，时间复杂度 O( logn )
             
                  显然在 index 的左边 为 全部小于 E 的元素，而在 index 的右边 全为 大于 E 的元素
            
            step4. 利用 预先计算好的 数组的 后缀和 ，直接拿到 index 的右边 的元素和 
            
            step5. 根据公式   E[k] = xk * E[k - 1] / N + Σi>xk (Vi / N) （其中 xk 为 index 的左边的元素个数） 更新 期望 E   
            
            整个算法的 时间复杂度： O( nlogn + Klogn)
            
        :param N: 5
        :param K: 3
        :param values: [16,11,7,4,1]
        :return: 12.3584
        """

        E = sum(values) / N  # 7.8

        #step1
        values = sorted(values)  # [1,4,7,11,16]

        #step2
        suffix_sum_cache=[0]*N

        suffix_sum_cache[-1]=values[-1]

        for i in range(N-2,-1,-1):

            suffix_sum_cache[i]= suffix_sum_cache[i+1]+values[i]

        #suffix_sum_cache=[39, 38, 34, 27, 16]

        for k in range(1, K + 1):  #

            # step3.
            index=self._find_insert_idx(values,E) # index=3

            if index==len(values) or index== 0: # 非法的 index 要特殊处理
                #eg. values=[80000,80000,80000] index=3 , 显然 E=80000

                break

            # left_set=values[:index] #[1,4,7] 一共 3 个 元素 index=3
            # right_set=values[index:] # [11,16]

            # step4.
            right_set_sum=suffix_sum_cache[index]

            left_set_sum=index*E

            # step5.
            sum_val= left_set_sum + right_set_sum

            E=sum_val/len(values)

        res = E

        # print(res) #TODO: 提交 时 注释

        return '{:.6f}'.format(res)  # {:.6f} 保留 6位小数

if __name__ == '__main__':

    sol=solutions()

    # sol.lucky_dip(5,3,[16,11,7,4,1])
    # sol.lucky_dip(3, 5, [8000, 8000, 8000])

    # IDE 测试 阶段：
    test=Test()
    test.test_small_dataset(sol.lucky_dip)

    test.test_large_dataset(sol.lucky_dip)

    # 提交 阶段：
    # T = int(input()) # 一共T个 测试数据
    #
    # for t in range(1, T + 1):
    #
    #     N_K = [int(i) for i in input().split(' ')]
    #
    #     N,K=N_K[0],N_K[1]
    #
    #     values=[int(i) for i in input().split(' ')]
    #
    #     res = sol.lucky_dip(N,K,values)
    #
    #     print('Case #{}: {}'.format(t, res))









