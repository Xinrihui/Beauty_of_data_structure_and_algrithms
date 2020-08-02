#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit
import numpy as np

import math

import random as rand

class solutions:

    def inversion_regions(self,nums):
        """
        排序算法 中，我们用有序度来表示一组数据的有序程度，用逆序度表示一组数据的无序程度。
        假设我们有 n 个数据，我们期望数据从小到大排列，那完全有序的数据的有序度就是 n(n-1)/2，逆序度等于 0；相反，倒序排列的数据的有序度就是 0，逆序度是 n(n-1)/2。
        
        M1:暴力枚举
        :param nums: [2,4,3,1,5,6]
        :return: [(2,1) ,(4,3) ,(4,1) ,(3,1)]
        """
        res=[]
        for i in range(len(nums)):
            left=nums[i]

            for j in range(i,len(nums)):
                right=nums[j]
                if left>right:
                    res.append((left,right))
        return len(res),res

    def inversion_regions_v2(self, nums):
        """
        M2: 分治法
        套用分治的思想来求数组 A 的逆序对个数。我们可以将数组分成前后两半 A1 和 A2，
        分别计算 A1 和 A2 的逆序对个数 K1 和 K2，然后再计算 A1 与 A2 之间的逆序对个数 K3。
        那数组 A 的逆序对个数就等于 K1+K2+K3。
        
        利用归并排序的思想：
        归并排序中有一个非常关键的操作，就是将两个有序的小数组，合并成一个有序的数组。
        每次合并操作，我们都计算逆序对个数，把这些计算出来的逆序对个数求和，就是这个数组的逆序对个数了。
        :param nums: 
        :return: 
        """
        N=len(nums)

        self.nums=nums

        self.res_region=[]

        self.inversion_num=0

        self.__merge_sort(0,N-1) # N-1 划重点！

        return self.inversion_num,self.nums

    def __merge_sort(self,left,right):

        if right>left: # 划重点！递归注意递归退出条件！

            #1.分解和递归：左半边有序 和 右半边有序
            mid = (right + left) // 2 # (right + left) 划重点！

            self.__merge_sort(left,mid)
            self.__merge_sort(mid+1,right) # mid+1 划重点！ left=0 right=1 mid=0 , self.__process(1 ,1)
            # self.__process(mid , right)  # left=0 right=1 mid=0 会永远陷入 self.__process(0 ,1) 的循环

            #2. 合并：把左半边有序和右半边有序合起来 ，整个序列有序
            self.__merge(left,mid,right)

    def __merge(self,left, mid , right):

        merge_cache=[]

        i = left
        j = mid+1

        while i <=mid and j<=right:

            if self.nums[i] > self.nums[j]:
                merge_cache.append(self.nums[j])
                j+=1

            else:
                self.inversion_num += j - (mid + 1)  # self.nums[i] 肯定比 self.nums[mid + 1:j+1] 的所有元素 都大
                                                     # eg. nums=[2,3,4,|1,5,6]  i=0 j=4 mid+1=3
                                                     #
                merge_cache.append(self.nums[i])

                i+=1

        if i <=mid :
            self.inversion_num += (right - mid)*(mid-i+1) # eg. nums=[2,3,4,|-2,-1,0] , [2,3,4] 的所有元素 比 [-2,-1,0] 都大 ，所以逆序对 一共有 3*3=9 个
            merge_cache=merge_cache+self.nums[i:mid+1]

        if j<=right:

            merge_cache=merge_cache+self.nums[j:right+1]

        # print('left:',left,'mid:',mid,'right:',right )
        # print('left part:', self.nums[left:mid+1],'right part:',self.nums[mid+1:right+1])
        # print('merge_cache:',merge_cache)
        # print('inversion_num:', self.inversion_num)

        self.nums[left:right+1]=merge_cache

    def max_subarray(self,nums):
        """
        最大子数组问题 （股票问题）
        
        :param nums: [-2,1,-3,4,-1,2,1,-5,4]
        :return: 
        """
        if len(nums)==1:
            return (nums[0],nums)


        mid=len(nums)//2

        # 1.分解 和 递归：最大子数组 在 mid 的 左边 或者 右边
        left_max_val,left_list=self.max_subarray(nums[:mid]) # 左半部分 的 最大子数组的值，和子数组
        right_max_val,right_list=self.max_subarray(nums[mid:])

        # 2. 中间状况：最大子数组 横跨了mid 的 左边和 右边的元素；时间复杂度 : O(n)

        mid_max_val_left = float('-inf') #最大子数组 的左半部分 的值
        left_row=mid-1

        mid_max_val_right = float('-inf') #最大子数组 的右半部分 的值
        right_row=mid

        # 找到 最大子数组 的左半部分 最大的情况
        left_sum=0
        i=mid-1 # mid >=1
        while i>=0:  # 时间复杂度 : O(n/2)
            left_sum +=nums[i]

            if left_sum>mid_max_val_left:
                mid_max_val_left= left_sum
                left_row= i

            i=i-1

        # 找到 最大子数组 的右半部分 最大的情况
        right_sum=0
        j=mid  # mid >=1
        while j < len(nums):# 时间复杂度 : O(n/2)
            right_sum +=nums[j]

            if right_sum>mid_max_val_right:
                mid_max_val_right= right_sum
                right_row= j

            j=j+1

        # 最大子数组 的左半部分 和 右半部分 拼一起
        mid_max_val=mid_max_val_left+mid_max_val_right
        mid_list=nums[left_row:right_row+1]

        result_list=[(left_max_val,left_list), (right_max_val,right_list),(mid_max_val,mid_list)]

        #3.合并：取 左边 右边 和中间 里的最优解
        max_result=max(result_list,key=lambda x: x[0])

        return max_result


class solution2:

    def min_dist_node_1D(self,nodes):
        """
        最接近点对问题：
        给定 线段 上的 n个点，找其中的一对点，使得在n个点的所有点对中，该点对的距离最小。
        
        M1: 线段上的点进行排序，用一次线性扫描就可以找出最接近点对
        :param nodes: [1,4,5,7,9]
        :return: 
        """
        nodes=sorted(nodes)
        prev_node=nodes[0]

        min_dist=float('inf')
        min_dist_nodes=None
        for i in range(1,len(nodes)):
            current_node= nodes[i]

            if current_node-prev_node < min_dist:


                min_dist=current_node-prev_node
                min_dist_nodes=(prev_node,current_node)

            prev_node=current_node

        return min_dist,min_dist_nodes

    def min_dist_node_1D_v2(self,nodes):
        """
        给定 线段 上的 n个点，找其中的一对点，使得在n个点的所有点对中，该点对的距离最小。
        
        M2 ： 分治法（要想快，用分治）
        ref: https://blog.csdn.net/liufeng_king/article/details/8484284
        :param nodes: [1,3,5,6,9]
        :return: 
        """
        if len(nodes)<=1: # 注意递归退出条件
            min_dist=float('inf')
            res_node=None

            return (min_dist,res_node)

        elif len(nodes)==2: # 注意考虑 递归到原子状况，即最小的子问题
            min_dist=abs(nodes[0]-nodes[1])
            res_node=(nodes[0],nodes[1])

            return (min_dist,res_node)


        min_dist = float('inf')
        res_node = None

        #1.分解：找到子问题 的切割方法，保证左右子问题规模近似
        m= (min(nodes)+max(nodes))/2 #找到切分点

        s1= list(filter(lambda t:t<=m, nodes)) # m的 左边的区间为 s1
        s2= list(filter(lambda t:t>m, nodes))  # m 的 右边区间为 s2
        # print(nodes_small)

        # 2. 递归：求解 s1 和 s2 中的 最接近点对
        min_dist_s1, s1_node  =self.min_dist_node_1D_v2(s1)
        min_dist_s2, s2_node = self.min_dist_node_1D_v2(s2)

        # 3. 中间情况： 横跨 s1 和 s2 的最接近点对
        # 4. 合并：三种情况（s1中的 最接近点对 ,s2中的 最接近点对，横跨 s1 和 s2 的最接近点对）中 找到最优解
        if min_dist_s1 < min_dist_s2:
            min_dist=min_dist_s1
            res_node=s1_node

        else:
            min_dist = min_dist_s2
            res_node = s2_node

        s1 = list(filter(lambda t: m-min_dist<t<=m , s1))
        s2 = list(filter(lambda t: m<=t<m+min_dist , s2))

        if len(s1)==1 and len(s2)==1:
            min_dist=abs(s1[0]-s2[0])
            res_node=(s1[0],s2[0])

        return (min_dist,res_node)

    def min_dist_node_2D(self, nodes):
        """
        给定 2D 平面 上的 n个点，找其中的一对点，使得在n个点的所有点对中，该点对的距离最小。
        
        M1: 枚举法 
        :param nodes: 
        :return: 
        """
        min_dist=float('inf')
        res_node=None

        for i in range(len(nodes)):
            left=nodes[i]

            for j in range(i+1,len(nodes)):
                right=nodes[j]

                # dist= math.sqrt((left[0]-right[0])**2+(left[1]-right[1])**2)
                dist = (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2

                if dist<min_dist:
                    min_dist=dist
                    res_node=(left,right)

        return min_dist,res_node


    def __partion(self,nums, left, right):

        pivot = rand.randint(left, right)

        nums[right], nums[pivot] = nums[pivot], nums[right]

        storeIndex = left

        for i in range(left, right):
            if nums[i] < nums[right]:
                nums[i], nums[storeIndex] = nums[storeIndex], nums[i]

                storeIndex += 1

        nums[right], nums[storeIndex] = nums[storeIndex], nums[right]
        return storeIndex

    def quick_select(self,nums, left, right, smallest_i):
        """
        基于 快速排序的 快速 选择，即找到 数组中第 N小的元素，若 N=len(nums)/2 即找到中位数
        平均时间复杂度：O(n)
        eg1.
        # l=[5,4,3,6,9,0]
        # target=quick_select( l, 0, len(l)-1, 0)  # smallest_i=0 即最小的元素 ，输出 0
        :param nums: 
        :param left: 左端点
        :param right: 右端点 
        :param smallest_i: 第 N小的 元素 
        :return: 
        """

        if left < right:

            pivot_new = self.__partion(nums, left, right)  # pivot_new is the index
            # print ("index:", pivot_new, "ele:", nums[pivot_new])
            # print ("list: ", nums)

            if pivot_new == smallest_i:
                return nums[pivot_new]
            elif pivot_new < smallest_i:  # smallest_i is the last position of the sorted list
                return self.quick_select(nums, pivot_new + 1, right, smallest_i)  # right
            else:
                return self.quick_select(nums, left, pivot_new - 1, smallest_i)  # left

        else:
            return nums[left]

    def __dist(self,node1,node2):
        """
        计算 两点的 距离的 平方
        :param node1: 
        :param node2: 
        :return: 
        """
        return  (node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2

    def min_dist_node_2D_v2(self, nodes):
        """
        给定 2D 平面 上的 n个点，找其中的一对点，使得在n个点的所有点对中，该点对的距离最小。
        
        M2 : 分治法
        :param nodes: 
        :return: 
        """
        if len(nodes)<=1: # 注意递归退出条件
            min_dist=float('inf')
            res_node=None

            return (min_dist,res_node)

        elif len(nodes)==2: # 注意考虑 递归到原子状况，即最小的子问题

            min_dist=self.__dist(nodes[0],nodes[1])
            res_node=(nodes[0],nodes[1])

            return (min_dist,res_node)

        nodes_x=list(map(lambda t:t[0],nodes))

        m=self.quick_select(nodes_x,0,len(nodes_x)-1, len(nodes_x)//2)
        # m= (min(nodes_x)+max(nodes_x))/2 #找到切分点


        s1= list(filter(lambda t:t[0]<=m, nodes)) # m的 左边的区间为 s1
        s2= list(filter(lambda t:t[0]>m, nodes))  # m 的 右边区间为 s2
        # print(nodes_small)

        min_dist_s1, s1_node  =self.min_dist_node_2D_v2(s1)
        min_dist_s2, s2_node = self.min_dist_node_2D_v2(s2)

        res_node=None

        if min_dist_s1 < min_dist_s2:
            min_dist=min_dist_s1
            res_node=s1_node

        else:
            min_dist = min_dist_s2
            res_node = s2_node

        d=min_dist

        # 用 P1和 P2 分别表示直线l的左边和右边的宽为 d 的2个垂直长条
        P1 = list(filter(lambda t: m - d < t[0] <= m, s1))
        P2 = list(filter(lambda t: m <= t[0] < m+d , s2))

        #将P1和P2中所有点按其y坐标排好序，则对P1中所有点 p，
        # 对排好序的点列作一次扫描，就可以找出所有最接近点对的候选者，对P1中每一点最多只要检查P2中排好序的 相继6个点，
        # 即 从P点的纵坐标yp 向上检查 P2中的三个点，向下检查 P2中的三个点。

        P_y_sorted=sorted(P1+P2,key=lambda t: t[1]) # 按照 纵坐标 对 P1,P2  进行排序
        P1=set(P1)
        P2=set(P2)

        for i,node in enumerate(P_y_sorted):

            if node in P1:
                # 向上找3个
                j=i
                up_times=0
                while j>=0:

                    if up_times>3:
                        break

                    if  P_y_sorted[j] in P2:
                       dist=self.__dist(node,P_y_sorted[j])
                       up_times+=1

                       if dist<min_dist:
                           min_dist = dist
                           res_node = (node,P_y_sorted[j])
                    j=j-1

                # 向下找3个
                j=i
                down_times=0
                while j < len(P_y_sorted):

                    if down_times>3:
                        break

                    if  P_y_sorted[j] in P2:
                       dist=self.__dist(node,P_y_sorted[j])
                       down_times+=1

                       if dist<min_dist:
                           min_dist = dist
                           res_node = (node,P_y_sorted[j])
                    j=j+1


        return (min_dist,res_node)


if __name__ == '__main__':

    sol = solutions()
    # print(sol.inversion_regions([2, 4, 3, 1, 5, 6]))
    # print(sol.inversion_regions_v2( [2, 4, 3, 1, 5, 6]))

    # inversion_regions 的大数据量 测试 ，统计运行时间
    # regions=np.random.randint(100,size=10000)
    #
    # regions=list(regions)

    # start = timeit.default_timer()
    # print('by inversion_regions: ')
    # print(sol.inversion_regions(regions)[0])
    # end = timeit.default_timer()
    # print('time: ', end - start, 's')
    #
    # start = timeit.default_timer()
    # print('by inversion_regions_v2: ')
    # print(sol.inversion_regions_v2(regions)[0])
    # end = timeit.default_timer()
    # print('time: ', end-start ,'s')

#-------------------------------------------------------------#
    nums=[-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(sol.max_subarray(nums))

#-------------------------------------------------------------#

    sol = solution2()
    # nodes_1D=10*(np.random.rand(100))
    nodes_1D=rand.sample(range(0,100),10) # [0,1000]中 抽样 100个不重复的数

    # nodes_1D=[1,3,5,6,9]
    nodes_1D=list(nodes_1D)
    # print(nodes_1D)


    # print(sol.min_dist_node_1D(nodes_1D))
    # print(sol.min_dist_node_1D_v2(nodes_1D))

    # x=rand.sample(range(0,100),10)
    # y=rand.sample(range(0,100),10)
    # nodes_2D=[(x[i],y[i]) for i in range(len(x))]

    # print(nodes_2D)
    # print(sol.min_dist_node_2D(nodes_2D))
    # print(sol.min_dist_node_2D_v2(nodes_2D))


    # min_dist_node_2D 的大数据量 测试 ，统计运行时间
    # x=rand.sample(range(0,10000),5000)
    # y=rand.sample(range(0,10000),5000)
    # nodes_2D=[(x[i],y[i]) for i in range(len(x))]
    #
    # start = timeit.default_timer()
    # print('by inversion_regions: ')
    # print(sol.min_dist_node_2D(nodes_2D))
    # end = timeit.default_timer()
    # print('time: ', end - start, 's')
    #
    # start = timeit.default_timer()
    # print('by inversion_regions_v2: ')
    # print(sol.min_dist_node_2D_v2(nodes_2D))
    # end = timeit.default_timer()
    # print('time: ', end-start ,'s')













