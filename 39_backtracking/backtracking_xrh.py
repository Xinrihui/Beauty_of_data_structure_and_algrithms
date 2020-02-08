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

    # for ele in permutations('ABC', 3):
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

    # for ele in combinations('ABC',2):
    #     print(ele) #  ('A', 'B')  ('A', 'C') ('B', 'C')



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



class solution_zero_one_bag_weight:

    def zero_one_bag_weight(self,weights, capacity):

        L = len(weights)

        max_bag_weight = 0

        res_bag = ()

        for l in range(1, L + 1):
            # 遍历 背包的 中的物品件数 l=1, l=2, l=3

            for bag in itertools.combinations(weights, l):  # 背包中的物品件数 为l 时，列举所有可能的物品的组合

                bag_weight = sum(map(lambda x: x[1], bag))  # 背包中 所有物品的重量求和
                # print('bag:', bag, ' weight: ', bag_weight)

                if bag_weight > max_bag_weight and bag_weight <= capacity:
                    max_bag_weight = bag_weight
                    res_bag = bag

        return max_bag_weight, res_bag

    def zero_one_bag_weight_recursive(self,weights,capacity):
        self.weights=weights
        self.capacity=capacity

        self.max_bag_weight=0
        self.res_bag=()

        current_bag=[]
        self.__process(-1,current_bag) #current_bag 表示当前已经装进去的物品；i表示考察到哪个物品了；

        return self.max_bag_weight,self.res_bag

    def __process(self,i,current_bag):

        if i < len(self.weights):

            bag_weight = sum(map(lambda x: x[1], current_bag))

            if bag_weight > self.capacity: #搜索剪枝: 当发现已经选择的物品的重量超过 Wkg 之后，我们就停止继续探测剩下的物品
                return

            print('bag:', current_bag, ' weight: ', bag_weight)

            if bag_weight >self.max_bag_weight:
                self.max_bag_weight=bag_weight
                self.res_bag = current_bag

            self.__process(i+1,current_bag) # 第i 个物品不放入背包
            self.__process(i + 1, current_bag+[self.weights[i]]) # 第i 个物品 放入背包



class solution_pattern_match:

    def match(self,pattern,text):

        self.pattern=pattern # 正则表达式
        self.text=text # 待匹配的字符串

        self.pattern_len=len(pattern)
        self.text_len=len(text)

        pattern_j=0
        text_i=0

        self.match_flag=False

        self.__process(text_i,pattern_j)

        return self.match_flag

    def __process(self,text_i,pattern_j):

        if self.match_flag == True: #如果已经匹配了，就不要继续递归了
            return

        if text_i==self.text_len:
            if pattern_j == self.pattern_len:
                # pattern 和 text 都到了末尾，说明模式匹配成功
                self.match_flag = True
                return

        if text_i<self.text_len and pattern_j< self.pattern_len: #保证数组不越界

            if self.pattern[pattern_j]=='*' :
                for index in range(text_i,self.text_len+1):  # 为了让指针 指向 text的末尾 ，self.text_len+1
                    # '*' 可以匹配任意个数的字符
                    #递归 检查 从text 的当前指针指向的位置 到 text 的末尾 与 pattern_j+1 的匹配
                    self.__process(index, pattern_j+1)

            elif self.pattern[pattern_j]=='?' :
                self.__process(text_i, pattern_j + 1) # 匹配0个字符
                self.__process(text_i+1, pattern_j + 1) #匹配1个字符

            else: # self.pattern[pattern_j] 为普通字符
                if self.text[text_i]==self.pattern[pattern_j]:
                    self.__process(text_i + 1, pattern_j + 1)

class solution_Traveling_Salesman_Problem:

    def tsp_recursive(self,city_distance,start_point):
        """
        递归+剪枝 解 旅行推销员问题
        ref: https://www.cnblogs.com/dddyyy/p/10084673.html
        :param city_distance: 城市的距离矩阵
        :param start_point:  起点城市
        :return: 
        """

        self.city_distance=city_distance
        self.city_names=list(range(len(city_distance)))

        self.start_point=start_point

        self.min_cost=float('inf') # 取一个足够大的数 作为初始的 最小路径代价
        self.min_cost_path=[]

        self.path_length=len(city_distance)+1 # path_length=5

        stage = 0  # stage0：把起始的点 放入 当前路径中
        current_path=[start_point]
        current_cost=0

        self.__process(stage+1,current_path,current_cost)

        return self.min_cost,self.min_cost_path

    def __process(self,stage,current_path,current_cost):

        if stage < self.path_length-1: # stage1-stage3

            if current_cost >= self.min_cost:
                return

            # print(current_path)
            for next_city in set(self.city_names)-set(current_path): # 每个城市只会访问一次，从没走过的城市中选一个访问

                current_city=current_path[-1]
                cost= current_cost + self.city_distance[current_city][next_city] # 路径代价为：当前的路径代价 + 现在所在城市到下一个城市的距离
                self.__process(stage + 1, current_path+[next_city],cost)

        elif stage==self.path_length-1: #stage4: 最后要回到 起点

            cost = current_cost + self.city_distance[current_path[-1]][self.start_point]
            current_path=current_path+[self.start_point] # 把 起始节点 加到路径的末尾
            if cost < self.min_cost: #
                self.min_cost=cost
                self.min_cost_path=current_path



class solution_Graph_Coloring_Problem():

    def mGCP(self,mapGraph,m):
        """
        递归 解 图的 m 着色 问题
        ref: https://blog.csdn.net/duyujian706709149/article/details/80581921
        :param mapGraph: 
        :param m: 
        :return: 
        """

        self.mapGraph=mapGraph
        self.colors= list(range(1,m+1))
        self.stageNum= len(mapGraph) # stageNum=5

        self.flag=False
        self.res_colors=[]

        #stage0: 初始节点 可以选择 所有颜色中的任意一种颜色上色
        stage=0
        current_colors=[]
        for color in self.colors:

            self._process( stage+1 ,current_colors+[color])

        return self.flag,self.res_colors

    def _process(self,stage,current_colors):

        if self.flag:
            return

        if stage < self.stageNum:  # stage1-stage4

            node=stage # 当前待上色的节点
            adjacent_nodes=self.mapGraph[node] #待上色节点的 相邻节点
            adjacent_nodes=adjacent_nodes[:len(current_colors)] #相邻节点中 截取 已经上过色的节点

            adjacent_nodes_colors=[ current_colors[index] for index,node in enumerate(adjacent_nodes) if node==1] #已经上过色的节点的颜色集合

            available_colors=set(self.colors)-set(adjacent_nodes_colors) #当前待上色的节点的可选颜色

            for color in available_colors:
                self._process(stage + 1, current_colors + [color])


        elif stage== self.stageNum: #stage5: 进入 该stage 说明 所有节点都已经成功上色
            self.flag=True
            self.res_colors=current_colors



if __name__ == '__main__':

    # test()
    # test1()
    # test2()

    sol = solution_zero_one_bag_weight()
    items_info = [('a',2),('b',2), ('c',4), ('d',6), ('e',7)]
    capacity = 9
    print(sol.zero_one_bag_weight(items_info, capacity))
    print(sol.zero_one_bag_weight_recursive(items_info, capacity))


    sol2=solution_pattern_match()
    pattern='a*d'
    text='abcd'
    # print(sol2.match(pattern,text))
    # print(sol2.match('a*d?f', 'abcdef'))
    # print(sol2.match('', 'ab'))
    # print(sol2.match('a*', 'ab'))
    # print(sol2.match('a?', 'ab'))
    # print(sol2.match('*', 'ab'))
    # print(sol2.match('?', 'a'))
    # print(sol2.match('**', 'ab'))
    # print(sol2.match('??', 'ab'))



    city_distance = [
        [0, 4, 3, 1],
        [4, 0, 1, 2],
        [3, 1, 0, 5],
        [1, 2, 5, 0],
    ]
    sol3=solution_Traveling_Salesman_Problem()
    # print(sol3.tsp_recursive(city_distance,0))


    mapGraph = [
        [0, 1, 1, 1, 0],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 0],
        [1, 1, 1, 0, 1],
        [0, 1, 0, 1, 0]
    ]
    sol4 = solution_Graph_Coloring_Problem()
    # print(sol4.mGCP(mapGraph, 4))
    # print(sol4.mGCP(mapGraph, 3))





