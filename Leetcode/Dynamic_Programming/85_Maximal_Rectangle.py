#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

import random

import numpy as np
class Solution:

    # maximalRectangle
    def solve_v1(self, matrix):
        """
        (TLE)
        
        run large dataset: 
        time:  0.39731681799999996 s
        
        时间复杂度: O(m*n*m)
        
        rows == matrix.length
        cols == matrix.length
        0 <= row, cols <= 200

        :param s: 
        :return: 
        """
        if len(matrix)==0:

            return 0

        matrix=np.array(matrix)

        # print(matrix)

        m,n=matrix.shape

        # m<n 好
        # if m>n: 若 m>n 则将矩阵进行 转置
        #     matrix=matrix.T
        #     m, n = matrix.shape

        dp_left = np.zeros((m + 1, n + 1), dtype='uint16')

        merge_row=np.zeros((m + 1, n + 1), dtype='uint16')


        for i in range(1,m+1):
            for j in range(1,n+1):

                matrix_i=i-1
                matrix_j=j-1

                if matrix[matrix_i][matrix_j]=='1':


                    dp_left[i][j] = dp_left[i][j-1] + 1

                    #1.按行进行合并
                    start_row=i

                    min_length=dp_left[i][j]
                    max_area = 0

                    rows_num=1

                    while dp_left[start_row][j]>0 and start_row>=0: # 向上找 TODO:时间复杂度 O(m) 需要优化

                        min_length=min(min_length,dp_left[start_row][j])

                        area=min_length*rows_num

                        max_area=max(area,max_area)

                        rows_num += 1

                        start_row-=1


                    merge_row[i][j]=max_area # TODO: 最大面积 可以做成 全局更新的


        # print(dp_left)
        # print(merge_row)

        res= np.max(merge_row)  # TODO: 少找一次 最大值

        return res

    def solve_v2(self, matrix):
        """
        在 solve_v1 基础上优化 常数量级的 时间复杂度

        run large dataset: 
        time:  0.03078084999999997 s

        时间复杂度: O(m*n*m)


        :param s: 
        :return: 
        """
        if len(matrix) == 0:
            return 0

        m, n = len(matrix),len(matrix[0])

        # dp_left = np.zeros((m + 1, n + 1),dtype=int) # TODO:  dtype='uint16' 才是 TLE 的 罪魁祸首

        dp_left =  [ [0] * (n+1) for _ in range(m+1)] # 快速 生成 m+1*n+1 矩阵 比 np.zeros() 快一个数量级

        max_area=0 # 全局更新

        for i in range(1, m + 1):
            for j in range(1, n + 1):

                matrix_i = i - 1
                matrix_j = j - 1

                if matrix[matrix_i][matrix_j] == '1':

                    dp_left[i][j] = dp_left[i][j - 1] + 1

                    # 1.按行进行合并
                    start_row = i

                    min_length = dp_left[i][j]

                    rows_num = 1

                    while dp_left[start_row][j] > 0 and start_row >= 0:  # 向上找 TODO:时间复杂度 O(m) 需要优化

                        min_length = min(min_length, dp_left[start_row][j])

                        area = min_length * rows_num

                        max_area = max(area, max_area)

                        rows_num += 1

                        start_row -= 1


        # print(dp_left)

        return max_area


    def get_max_area(self,heights):
        """
        直方图 中能画出的 最大面积的 矩形
        
        时间复杂度: O(N) 
        
        :param heights: 
        :return: 
        """
        # heights=list(heights)

        heights.insert(0,0) # 首尾 补0
        heights.append(0)

        stack=[]

        stack.append( (0,heights[0]) )

        areas=[0]*len(heights) # 高度为 heights[i] 的矩形的面积

        for i in range(1,len(heights)):

            if heights[i]>= stack[-1][1]: # heights[i] 比栈顶元素大 则入栈

                stack.append((i,heights[i]))

            else : # heights[i] < stack[-1]

                while heights[i]<stack[-1][1]: # heights[i] 比栈顶元素小 栈弹出栈顶元素

                    index,h=stack.pop()

                    stack_top_ele=stack[-1]

                    areas[index]=(i-stack_top_ele[0]-1)*h

                stack.append((i, heights[i]))

        # print(areas)

        return max(areas)




    def solve_v3(self, matrix):
        """
        优化 时间复杂度 至: O(m*n)
        
        run large dataset: 
        time:  0.03354396799999998 s

        :param s: 
        :return: 
        """

        if len(matrix) == 0:
            return 0


        m, n = len(matrix), len(matrix[0])

        # dp_left = np.zeros((m + 1, n + 1), dtype=int)

        dp_left = [[0] * (n + 1) for _ in range(m + 1)]  # 快速 生成 m+1*n+1 矩阵 比 np.zeros() 快一个数量级

        max_area = 0  # 全局更新

        for i in range(1, m + 1):
            for j in range(1, n + 1):

                matrix_i = i - 1
                matrix_j = j - 1

                if matrix[matrix_i][matrix_j] == '1':

                    dp_left[i][j] = dp_left[i][j - 1] + 1

        dp_left=np.array(dp_left) # 转换为 np.array 方便切片

        for j in range(1, n + 1):

            max_area=max(max_area,self.get_max_area(list(dp_left[:,j]))) # dp_left[:,j] 取 2D数组的 第j列

        return max_area



class Test:
    def test_small_dataset(self, func):

        m=[["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
        assert func(m) == 6

        m = [["1", "0", "1", "1", "0"],
             ["1", "0", "1", "1", "1"],
             ["1", "1", "1", "1", "1"],
             ["1", "0", "1", "1", "1"]]
        assert func(m) == 9

        m = [["1", "1", "1", "1"], ["1", "1", "1", "1"], ["1", "1", "1", "1"]]
        assert func(m) == 12

        m = [["1", "0", "1", "1", "0"],
             ["1", "0", "1", "1", "1"],
             ["1", "1", "1", "1", "1"],
             ["1", "1", "1", "1", "1"]]
        assert func(m) == 10

        m=[["1", "1", "1", "1", "1", "1", "1", "1"],
           ["1", "1", "1", "1", "1", "1", "1", "0"],
           ["1", "1", "1", "1", "1", "1", "1", "0"],
           ["1", "1", "1", "1", "1", "0", "0", "0"],
           ["0", "1", "1", "1", "1", "0", "0", "0"]]
        assert func(m) == 21

        m = [["0", "1", "1", "0", "1"],
             ["1", "1", "0", "1", "0"],
             ["0", "1", "1", "1", "0"],
             ["1", "1", "1", "1", "0"],
             ["1", "1", "1", "1", "1"],
             ["0", "0", "0", "0", "0"]]
        assert func(m) == 9

        assert func([["0"]]) == 0

        assert func([["1"]]) == 1

        assert func([["0","0"]]) == 0

        # TODO: 边界条件
        assert func( []) == 0
        #
        # assert func('') == ''

    def test_large_dataset(self, func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题

        Limits
        
        rows == matrix.length
        cols == matrix.length
        0 <= row, cols <= 200

        :param func: 
        :return: 
        """

        row_num=200
        cols_num=200

        matrix = [[ random.choices(['0', '1'])[0] for __ in range(row_num)] for __ in range(cols_num)]

        start = timeit.default_timer()
        print('run large dataset: ')
        func(matrix)
        end = timeit.default_timer()
        print('time: ', end - start, 's')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：


    #m=[["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]

    # m = [["1", "0", "1", "1", "0"],
    #      ["1", "0", "1", "1", "1"],
    #      ["1", "1", "1", "1", "1"],
    #      ["1", "0", "1", "1", "1"]]

    # m=[["1", "1", "1", "1"], ["1", "1", "1", "1"], ["1", "1", "1", "1"]]


    # m=[["1", "1", "1", "1", "1", "1", "1", "1"],
    #    ["1", "1", "1", "1", "1", "1", "1", "0"],
    #    ["1", "1", "1", "1", "1", "1", "1", "0"],
    #    ["1", "1", "1", "1", "1", "0", "0", "0"],
    #    ["0", "1", "1", "1", "1", "0", "0", "0"]]

    # m = [["1", "0", "1", "1", "0"],
    #      ["1", "0", "1", "1", "1"],
    #      ["1", "1", "1", "1", "1"],
    #      ["1", "1", "1", "1", "1"]]

    m=[["0", "1", "1", "0", "1"],
       ["1", "1", "0", "1", "0"],
       ["0", "1", "1", "1", "0"],
       ["1", "1", "1", "1", "0"],
       ["1", "1", "1", "1", "1"],
       ["0", "0", "0", "0", "0"]]

    # print(sol.solve(m))

    # print(sol.get_max_area([2,1,5,6,2,3]))

    # IDE 测试 阶段：
    test = Test()
    # test.test_small_dataset(sol.solve_v1)
    # test.test_large_dataset(sol.solve_v1)
    #
    test.test_small_dataset(sol.solve_v2)
    test.test_large_dataset(sol.solve_v2)
    #
    test.test_small_dataset(sol.solve_v3)
    test.test_large_dataset(sol.solve_v3)










