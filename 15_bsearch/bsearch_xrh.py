
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit
import numpy as np

import sys
import random as rand



class solutions:

    def bsearch(self, nums,target):
        """
        二分查找 
        :param nums: 
        :param target: 
        :return: 
        """

        nums=sorted(nums)

        left=0
        right=len(nums)-1

        while left<=right:

            mid= left+(right-left)//2 #防止 left+right 过大溢出

            if nums[mid]==target:
                return mid # 返回下标

            elif target>nums[mid]:
                left=mid+1

            elif target<nums[mid]:
                right=mid-1

        return None

    def sqrt(self,a):
        """
        求一个数的平方根 ,要求精确到小数点后 6 位
        
        分治法
        :param a: 
        :return: 
        """
        left = 0
        right=float(a)

        while right-left >= 1.0e-6:

            mid= (right+left)/2.0

            if pow(mid,2)==a:
                return mid

            elif a>pow(mid,2):

                left = mid

            elif a<pow(mid,2):
                right = mid

        return left


if __name__ == '__main__':

    sol = solutions()

    l=np.random.randint(int(1e7),size=int(1e7)) # 1000 万个整数
    l1=list(l)
    # print(l1)
    print('l1 list memory_size:', sys.getsizeof(l1),'B') # 90000112 B= 90MB

    l2=set(l)
    print('l2 hash memory_size:', sys.getsizeof(l2),'B') #  268435680 B= 268MB

    start = timeit.default_timer()
    print('by bsearch: ')
    print(sol.bsearch(l1, 100)) # 在l1 中查找100
    end = timeit.default_timer()
    print('time: ', end-start ,'s')

    start = timeit.default_timer()
    print('by hash search: ')
    print( 100 in l2 )
    end = timeit.default_timer()
    print('time: ', end-start ,'s')


    # print(pow(9,0.5),sol.sqrt(9))
    # print(pow(4, 0.5), sol.sqrt(4))
    # print(pow(10, 0.5), sol.sqrt(10))





