#!/usr/bin/python
# -*- coding: UTF-8 -*-

import itertools

def test():
    """
    利用 itertools 实现 排列 和 组合
    ref https://www.cnblogs.com/xiao-apple36/p/10861830.html
    :return: 
    """

    #1.组合
    for i in itertools.combinations('ABC', 1):
        print(i)

    for i in itertools.combinations('ABC', 2):
        print(i)
    # 输出 AB AC BC

    for i in itertools.combinations('ABC', 3):
        print(i)
    # 输出  ABC


    #2.排列
    for i in itertools.permutations('ABC', 2):
        print(''.join(i), end=",")
    # 输出 BC BD CB CD DB DC
    print('\r')


    #3. 笛卡尔积
    a = (1, 2)
    b = ('A', 'B', 'C')
    c = itertools.product(a, b)
    for i in c:
        print(i)

    for i in itertools.product('ABC', repeat=2): # a='ABC' b='ABC' a与b 做笛卡尔积
        print(''.join(i), end=",")
    print('\n')

def test1():
    """
    纯手工 实现 排列 与 组合 
    ref: https://docs.python.org/zh-cn/3.7/library/itertools.html
    
    :return: 
    """
    # 1. 笛卡尔积
    def product(*args, repeat=1):
        """
        eg.
         product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
         product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111        
        :param args: 
        :param repeat: 
        :return: 
        """

        pools = [tuple(pool) for pool in args] * repeat # [('A', 'B', 'C'), ('A', 'B', 'C')]
        result = [[]]

        # for pool in pools:
        #     result = [x + [y] for x in result for y in pool]

        for pool in pools:
            res=[]
            for y in pool:
                for x in result:
                    res.append(x + [y])
            # print(res) #  [['A'], ['B'], ['C']] ;
                         #  [['A', 'A'], ['B', 'A'], ['C', 'A'], ['A', 'B'], ['B', 'B'], ['C', 'B'], ['A', 'C'], ['B', 'C'], ['C', 'C']]
            result=res


        for prod in result:
            yield tuple(prod)

    # for ele in product('ABC', repeat=2):
    #     print(ele)

    # 2. 排列
    def permutations(iterable, r=None):
        """
        permutations() 可被改写为 product() 的子序列，
        只要将含有重复元素（来自输入中同一位置的）的项排除。
        :param iterable: 
        :param r: 
        :return: 
        """
        pool = tuple(iterable)
        n = len(pool)
        r = n if r is None else r
        for indices in product(range(n), repeat=r):

            if len(set(indices)) == r: # len(set( ('A','A')))=1
                yield tuple(pool[i] for i in indices)

    # for ele in permutations('ABC', 2):
    #     print(ele)


    # 3. 组合
    def combinations(iterable, r):
        """
        combinations() 被改写为 permutations() 过滤后的子序列，（相对于元素在输入中的位置）元素不是有序的。
        :param iterable: 
        :param r: 
        :return: 
        """
        pool = tuple(iterable)
        n = len(pool)
        for indices in permutations(range(n), r):

            if sorted(indices) == list(indices):
                yield tuple(pool[i] for i in indices)

    def combinations_v2(iterable, r):
        """
        eg.
        combinations('ABCD', 2) --> AB AC AD BC BD CD
        combinations(range(4), 3) --> 012 013 023 123
        
        :param iterable: 
        :param r: 
        :return: 
        """

        pool = tuple(iterable) # ('A', 'B', 'C')
        n = len(pool) # n=3
        if r > n: # r=2
            return
        indices = list(range(r)) # indices=[0,1]

        yield tuple(pool[i] for i in indices) # (pool[0],pol[1])

        while True:
            for i in reversed(range(r)): # i=1
                                         # i=0 ;

                                         # i=1
                if indices[i] != i + n - r: # i + n - r: 1+3-2 =2 ;
                                            #            0+3-2=1
                    break
            else:
                return
            indices[i] += 1 # indices=[0,2] ;
                            # indices=[1,2] ;
                            #
            for j in range(i + 1, r):
                indices[j] = indices[j - 1] + 1
            yield tuple(pool[i] for i in indices) # (pool[0],pol[2]) ;
                                                  # (pool[1],pol[2])

    for ele in combinations('ABC',2):
        print(ele) #  ('A', 'B')  ('A', 'C') ('B', 'C')



from functools import reduce

def test2():
    """
    reduce 函数的使用
   reduce把一个函数作用在一个序列[x1, x2, x3, ...]上，这个函数必须接收两个参数，reduce把结果继续和序列的下一个元素做累积计算，其效果就是：
   reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
   
   ref: https://www.liaoxuefeng.com/wiki/1016959663602400/1017329367486080
    :return: 
    """

    def add(x, y):
        return x + y

    print(reduce(add, [1, 3, 5, 7, 9]))

    def fn(x, y):
        return x * 10 + y

    print(reduce(fn, [1, 3, 5, 7, 9])) #把序列[1, 3, 5, 7, 9]变换成整数13579


    DIGITS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

    def char2num(s):
        return DIGITS[s]

    def str2int(s):
        return reduce(lambda x, y: x * 10 + y, map(char2num, s))

    print(str2int('123'))




def zero_one_bag_weight(weights,capacity):

    L=len(weights)

    max_bag_weight=0

    res_bag=()

    for l in range(1,L+1):
        # 遍历 背包的 中的物品件数 l=1, l=2, l=3

        for bag in itertools.combinations(weights, l): # 背包中的物品件数 为l 时，列举所有可能的物品的组合

            bag_weight=sum( map( lambda x:x[1] , bag ) ) # 背包中 所有物品的重量求和
            print('bag:',bag,' weight: ',bag_weight)

            if bag_weight>max_bag_weight and bag_weight <= capacity:
                max_bag_weight=bag_weight
                res_bag=bag

    return max_bag_weight,res_bag

class solution:

    def zero_one_bag_weight_recursive(self,weights,capacity):
        self.weights=weights
        self.capacity=capacity

        self.max_bag_weight=0
        self.res_bag=()

        current_bag=[]
        self.__process(-1,current_bag)

        return self.max_bag_weight,self.res_bag

    def __process(self,i,current_bag):

        if i < len(self.weights):

            bag_weight = sum(map(lambda x: x[1], current_bag))

            if bag_weight > self.capacity:
                return

            # print(current_bag)

            if bag_weight >self.max_bag_weight:
                self.max_bag_weight=bag_weight
                self.res_bag = current_bag

            self.__process(i+1,current_bag) # 第i 个物品不放入背包
            self.__process(i + 1, current_bag+[self.weights[i]]) # 第i 个物品 放入背包



if __name__ == '__main__':

    # test()
    # test1()
    # test2()

    items_info = [('a',2),('b',2), ('c',4), ('d',6), ('e',7)]
    capacity = 14
    print(zero_one_bag_weight(items_info, capacity))

    sol=solution()
    print(sol.zero_one_bag_weight_recursive(items_info, capacity))

