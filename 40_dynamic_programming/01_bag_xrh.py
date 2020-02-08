#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import *

class solution_zero_one_bag_weight:

    def run_recursive_cache(self,weights,capacity):
        """
        回溯法 解 01背包问题 , 并缓存子问题的解
        ref: https://time.geekbang.org/column/article/74788 
        :param weights: 
        :param capacity: 
        :return: 
        """
        self.weights=weights
        self.capacity=capacity

        self.max_bag_weight=0
        # self.res_bag=()

        self.cache = zeros((len(weights) , capacity+1), dtype=bool)

        # print(self.cache)

        i=-1
        current_bag_weight=0
        self.__f( i, current_bag_weight)

        return self.max_bag_weight

    def __f(self, i , current_bag_weight):

        if i < len(self.weights): # i= -1,0,1,2,3,4

            if current_bag_weight > self.capacity: #搜索剪枝: 当发现已经选择的物品的重量超过 Wkg 之后，我们就停止继续探测剩下的物品
                return

            # if self.cache[i][current_bag_weight] is True:
            #     return
            # else:
            #     self.cache[i][current_bag_weight] = True

            print(i, current_bag_weight)

            if current_bag_weight > self.max_bag_weight:
                self.max_bag_weight =current_bag_weight

            self.__f(i+1,current_bag_weight) # f(0,0)

            self.__f(i+1, current_bag_weight + self.weights[i] ) #f(0,2)



if __name__ == '__main__':

    # weight=[2,2,4,6,7]
    weight = [2,2]
    sol = solution_zero_one_bag_weight()
    print(sol.run_recursive_cache(weight,9))
    # print(sol.run_recursive_cache(weight, 21))




