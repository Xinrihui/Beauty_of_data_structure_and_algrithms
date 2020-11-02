#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

import numpy as np


import warnings
import functools

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


class Solution:
    """
    leetcode 64 题的变形（ MSRA 面试）
    https://leetcode-cn.com/problems/minimum-path-sum 
    
    类似的题目有
    leetcode 741
    https://leetcode-cn.com/problems/cherry-pickup/
    
    给定一个包含非负整数的 m x n 网格，
    请找出一条从左上角到右下角的路径（每次只能向下或者向右移动一步），使得路径上的数字总和为最大。
    
    在这基础上，假设有两个小人 同时从左上角走到右下角，求他们两个走的路径的和最大的走法，
    若他们两个走到 同样的格子中，则只能将 一个格子的值记录 路径的和
    

    示例:
    
    输入:
    [
      [1,2,1,5],
      [3,3,4,3],
      [2,2,1,2]
    ]
    输出: 
    
    小人1 走的路径 1->3->3->4->3->2  pathsum1=16
    小人2 走的路径 1(重复)->2->1->5->3(重复)->2(重复)  pathsum2=8
    
    两条路径的和 pathsum=pathsum1 + pathsum2=24
    
    """

    def __all_paths_recursion(self,i,j,prev_path,prev_path_sum):
        """
        递归 找出 所有的路径
        
        :param i: 
        :param j: 
        :param prev_path: 
        :param prev_path_sum: 
        :return: 
        """

        # print(i,j)

        if len(prev_path) == self.rowNum + self.colNum-1: # 每一条路径 走过的格子的数量是 固定的

            path= prev_path
            # print(path)
            path_sum=prev_path_sum

            self.path_list.append((path,path_sum))

            return

        if i < self.rowNum and j < self.colNum:

            self.__all_paths_recursion(i + 1, j, prev_path + [(i, j)], prev_path_sum + self.grid[i][j])

            self.__all_paths_recursion(i, j + 1, prev_path + [(i, j)], prev_path_sum + self.grid[i][j])

        elif i < self.rowNum and j == self.colNum-1: # 走到了 右边界

            # 只能往下走
            self.__all_paths_recursion(i+1,j,prev_path+[(i,j)],prev_path_sum+self.grid[i][j])


        elif j < self.colNum and i == self.rowNum-1: # 走到了 下边界

            # 只能往右走
            self.__all_paths_recursion(i,j+1,prev_path+[(i,j)],prev_path_sum+self.grid[i][j])



    def traverse_all_paths(self,grid):
        """
        
        遍历所有的路径 (回溯法)
        
        :param grid: 
        :return: 
        """
        self.grid=grid

        self.rowNum=len(grid)
        self.colNum=len(grid[0])

        i=0
        j=0
        prev_path=[]
        prev_path_sum=0

        self.path_list=[]
        self.__all_paths_recursion(i,j,prev_path,prev_path_sum)

        return self.path_list


    def __max_sum_path(self,grid):
        """
        动态规划 找一条 和最大的路径
        :param grid: 
        :return: 
        """

        rowNum=len(grid)

        colNum=len(grid[0])

        states= np.zeros((rowNum,colNum),dtype='int')

        # 边界情况处理
        states[0][0]=grid[0][0]

        for i in range(1,colNum):
            states[0][i]= states[0][i-1]+grid[0][i]

        for i in range(1,rowNum):
            states[i][0]= states[i-1][0]+grid[i][0]

        # 递推公式
        for i in range(1, rowNum):
            for j in range(1, colNum):

                states[i][j]=max(states[i-1][j], states[i][j-1])+grid[i][j]


        max_path_sum=states[rowNum-1][colNum-1] # 最大和的路径 的路径长度

        # 从终点 反向推理出 路径

        max_path=[]

        i=rowNum-1
        j=colNum-1

        while i>=0 and j>=0:

            max_path.append((i, j))

            if i-1>=0 and j-1>=0: # 防止越界

                if states[i-1][j] >= states[i][j-1]:
                    i=i-1
                else:
                    j=j-1
            else: #  i-1 <0  or  j-1<0
                if i-1 <0:
                    j = j - 1
                elif j-1<0:
                    i = i - 1

        return max_path_sum,max_path[::-1]

    def solve_naive(self, grid):
        """
        
        回溯法 解 最长路径
        
        1.递归得到 所有的路径 并计算路径的和
        
        2.遍历这些路径，对每一条 路径1：
           2.1 把这一条路径经过的格子置为 0, 再找出 路径和最大的 路径2
           2.2 计算 两条路径(路径1 和 路径2) 的 和
        
        3.找到 两条路径的和最大的 路径的组合
        
        :param grid: 
        :return: 
        """
        # s is None or s==''
        if not grid:
            return

        grid= np.array(grid)

        path_list=self.traverse_all_paths(grid)

        max_path_sum=0
        max_path_cp=[]

        for path1,path1_sum in path_list:

            grid_new=grid.copy()

            # 把路径上的 格子置为0
            for ele in path1:
                grid_new[ele[0]][ele[1]] = 0

            max_path_sum2, max_path2 = self.__max_sum_path(grid_new)

            if path1_sum+max_path_sum2 >max_path_sum:

                max_path_sum=path1_sum+max_path_sum2

                max_path_cp=((path1,path1_sum),(max_path2,max_path_sum2),max_path_sum)


        return max_path_cp


    # maxPathSum
    def solve_greedy(self, grid):
        """
        贪心法 解 最长路径
        
        TODO: 找不到 两条路径 整体的 最优路径的组合

        时间复杂度 O(m^n)

        :param s: 
        :return: 
        """
        # s is None or s==''
        if not grid:
            return

        grid=np.array(grid)

        # 1. 找到第一条最大的路径
        max_path_sum1, max_path1=self.__max_sum_path(grid)

        # print(max_path_sum, max_path)

        # 2. 把第一条路径上的 格子置为0

        grid_new = grid.copy()

        for ele in max_path1:
            grid_new[ele[0]][ele[1]]=0


        # 2. 接着找 第2条 和最大的路径
        max_path_sum2, max_path2 = self.__max_sum_path(grid_new)

        max_path_sum=max_path_sum1+max_path_sum2

        # print(max_path2)

        return ((max_path1,max_path_sum1),(max_path2,max_path_sum2),max_path_sum)

    def solve_dp(self, grid):
        """
        2路动态规划问题
        
        利用 4阶 dp 求解
        
        :param grid: 
        :return: 
        """
        rowNum=len(grid)+1
        colNum=len(grid[0])+1

        d= np.zeros((rowNum,colNum,rowNum,colNum),dtype='int') # rowNum=len(grid)+1 边界情况的处理技巧

        for i in range(1,rowNum):
            for j in range(1, colNum):
                for k in range(1, rowNum):
                    for g in range(1, colNum):

                        d_1= max(d[i-1][j][k-1][g],d[i][j-1][k][g-1],d[i-1][j][k][g-1],d[i][j-1][k-1][g])

                        if i==k and j==g: # A 点和 B点重合

                            d_2=grid[i-1][j-1] # grid[i-1][j-1] 边界情况的处理技巧

                        else:

                            d_2 = grid[i - 1][j - 1] + grid[k - 1][g - 1]

                        d[i][j][k][g] =d_1 + d_2


        max_path_sum=d[-1][-1][-1][-1]

        return max_path_sum

    def solve_dp_opt(self, grid):
        """
        2路动态规划 问题
        
        由4阶dp 优化为 3阶 dp 求解

        :param grid: 
        :return: 
        """
        m = len(grid) # rowNum
        n = len(grid[0]) # colNum

        L= m + n- 1

        # 1.划分子问题
        d = np.zeros((L+1, m+1, m+1), dtype='int')  # len(grid)+1 边界情况的处理技巧

        # 2. 子问题的 计算顺序
        for l in range(1, L+1):

            start= max(l-len(grid[0])+1,1)
            end=min(m,l)

            for i in range(start, end+1):
                for k in range(start, end+1):

                    # print(l,i,k)


                    d_1 = max(d[l - 1][i][k], d[l - 1][i-1][k-1],
                              d[l - 1][i-1][k],d[l - 1][i][k-1])

                    if i == k :  # A 点和 B点重合

                        d_2 = grid[i - 1][(l+1-i)-1]  #  边界情况的处理技巧

                    else:

                        d_2 = grid[i - 1][(l+1-i)-1] + grid[k - 1][(l+1-k)-1]

                    d[l][i][k] = d_1 + d_2

        print(d)
        max_path_sum = d[-1][-1][-1]

        # 追踪解：从终点 反向推理出 路径

        l=m + n- 1
        i=m
        k=m

        path1=[]
        path2=[]

        while l >=1:

            path1.append((i,l + 1 - i))
            path2.append((k, l + 1 - k))

            l-=1

            # i
            v1 = np.max(d[l][i][:i+1])
            idx1 = np.argmax(d[l][i][:i+1])

            # i-1
            v2 = np.max(d[l][i - 1][:i])
            idx2 = np.argmax(d[l][i - 1][:i])

            if v1 > v2:
                # i=i
                k=idx1

            else:
                i-=1
                k=idx2


        path1=path1[::-1]
        path2=path2[::-1]


        return path1,path2,max_path_sum





class Test:
    def test_small_dataset(self, func):


        # TODO: 边界条件
        assert func(None) == None

        assert func('') == ''

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

    grid=[
    [1,2,1,5],
    [3,3,4,3],
    [2,2,1,2]
    ]

    # print(sol.solve_greedy(grid))

    # print(sol.traverse_all_paths(grid))

    print(sol.solve_naive(grid))

    print(sol.solve_dp_opt(grid))

    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










