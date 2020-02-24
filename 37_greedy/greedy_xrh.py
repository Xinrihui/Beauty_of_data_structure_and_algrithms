#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import *
import heapq

class solutions:

    def childs_with_sugers(self,childs,sugers):
        """
        分糖给小朋友，一个小朋友只能拿一块糖，糖不能分割
        :param childs: 
        :param sugers: 
        :return: 
        """
        childs=sorted(childs)
        sugers=sorted(sugers)

        res=[]
        # if len(childs) > len(sugers) :
        j=0
        for i,child in enumerate(childs):

           while j < len(sugers):
               if sugers[j] >= child:
                    res.append([child,sugers[j]])
                    j+=1
                    break
               j=j+1
           else: # 正常结束 while 循环 则运行下面代码
               print('suger is not enough') # 糖已经分配完了，有的小朋友没有糖吃
               break # 跳出for 循环

        return res

    def regions_overlap(self,regions,L):
        """
        给定一个长度为 m的区间，再给出 n条线段的起点和终点（注意这里是闭区间），
        求最少使用多少条线段可以将整个区间完全覆盖。
        
        ref: https://www.cnblogs.com/acgoto/p/9824723.html
        :param regions: [ [2,6],[1,4],[3,6],[3,7],[6,8],[2,4],[3,5] ]
        :param L:  8 
        :return: [ [1,4] ,[3,7],[6,8] ]
        """
        regions=sorted(regions, key=lambda d: d[0]) # 按照区间的 左端点进行排序

        # print(regions)

        right_most=1
        res=[]

        while right_most<L:

            left_small=list(filter(lambda x: x[0] <= right_most, regions)) # 过滤出左端点 小于 right_most 的区间

            right_max=max(left_small,key=lambda x:x[1]) # 选这些区间 中 右端点最大的一个

            res.append(right_max)

            right_most=right_max[1] #更新 已覆盖线段的 右端点

        return res

    def regions_overlap2(self, regions, L):
        """
        假设我们有 n 个区间，区间的起始端点和结束端点分别是[l1, r1]，[l2, r2]，[l3, r3]，……，[ln, rn]。我们从这 n 个区间中选出一部分区间，
        这部分区间满足两两不相交（端点相交的情况不算相交），最多能选出多少个区间呢？
        
        ref: https://time.geekbang.org/column/article/73188
        :param regions: [[6,8],[2,4],[3,5],[1,5],[5,9],[8,10]]
        :param L: 10
        :return: [[2,4],[6,8],[8,10]]
        """
        regions = sorted(regions, key=lambda d: d[0])  # 按照区间的 左端点进行排序
        # print(regions)

        right_most=0
        res=[]

        while right_most<L:

            left_small=list(filter(lambda x: x[0] >= right_most, regions)) # 过滤出左端点 大于 right_most 的区间 ，这样能避免重合

            right_max=min(left_small,key=lambda x:x[1]) # 选这些区间 中 右端点最小的一个，这样 能留出更多的剩余空间

            res.append(right_max)

            right_most=right_max[1] #更新 已覆盖线段的 右端点

        return res


class ComapreHeap(object):
    def __init__(self, initial=None, key=lambda x: x):
        self.key = key
        if initial:
            self._data = [(key(item), item) for item in initial]
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, (self.key(item), item))

    def pop(self):
        return heapq.heappop(self._data)[1]

class TreeNode(object):
    def __init__(self,key=None,value=None):
        self.key=key
        self.value=value
        self.left=None
        self.right=None

class huffman_tree:

    def __init__(self, char_list):

        self.huffman_encode_tree=self.__encode(char_list)

    def decode_all(self):
        """
        返回 霍夫曼 编码树 上 char_list 中所有字符的 编码
        :return: 
        """
        root=self.huffman_encode_tree
        self.res={}

        self.__tree_pre_order(root,[])

        return self.res

    def __tree_pre_order(self,root,pre_list):

        if root.left== None and root.right == None: # 说明到达叶子节点
            self.res[root.key]=pre_list

        else:
           if root.left !=None:
               self.__tree_pre_order(root.left,pre_list+[0])
           if root.right!=None:
               self.__tree_pre_order(root.right, pre_list + [1])


    def __encode(self,char_list):
        """
        霍夫曼编码
        
        将 char_list 中的字符，根据其出现的频率 生成一颗 huffman 编码树 
        
        ref: 
        （1）《算法导论》
        （2）https://time.geekbang.org/column/article/73188
        :param char_list: [('a',45),('b',13),('c',12),('d',16),('e',9),('f',5)]
        :return: 
        """

        leaf_nodes= [TreeNode(ele[0] ,ele[1]) for ele in char_list ]

        heap_nodes=ComapreHeap(leaf_nodes,key=lambda x:x.value)

        N=len(char_list)

        root_node=None

        for i in range(N-1): # N 为叶节点个数，要执行N-1次的 叶节点的合并操作

            root_node=TreeNode()

            left_node=heap_nodes.pop()
            right_node=heap_nodes.pop()

            root_node.key='s'+str(i)
            root_node.value=left_node.value+ right_node.value

            root_node.left=left_node
            root_node.right=right_node

            print('root:',root_node.value)
            print('root.left:', root_node.left.value)
            print('root.right:', root_node.right.value)

            heap_nodes.push(root_node)


        return root_node



if __name__ == '__main__':

    sol = solutions()
    childs=[3,4,5,6,7,8] # 小孩0-小孩5 想要的糖果的 重量
    sugers=[1,2,3,4,5] # 现有的各个糖果的 重量

    # print(sol.childs_with_sugers(childs,sugers))

    regions=[ [2,6],[1,4],[3,6],[3,7],[6,8],[2,4],[3,5] ]
    L=8
    # print(sol.regions_overlap(regions,L))

    regions=[[6,8],[2,4],[3,5],[1,5],[5,9],[8,10]]
    L=10
    # print(sol.regions_overlap2(regions, L))

    char_list=[('a',45),('b',13),('c',12),('d',16),('e',9),('f',5)]
    huffman_tree=huffman_tree(char_list)
    print(huffman_tree.decode_all())


