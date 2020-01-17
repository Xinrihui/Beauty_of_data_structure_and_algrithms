#!/usr/bin/python
# -*- coding: UTF-8 -*-

# B Tree
# http://dieslrae.iteye.com/blog/1453853
# http://blog.csdn.net/v_july_v/article/details/6530142

class Entity(object):
    '''''数据实体'''

    def __init__(self, key, value):
        self.key = key
        self.value = value


class Node(object):
    '''''B树的节点'''

    def __init__(self):
        self.parent = None
        self.entitys = []
        self.childs = []

    def find(self, key):
        '''''通过key查找并返回一个数据实体'''



    def delete(self, key):
        ''''' 通过key删除一个数据实体,并返回它和它的下标(下标,实体) '''


    def isLeaf(self):
        '''''判断该节点是否是一个叶子节点'''



    def addEntity(self, entity):
        '''''添加一个数据实体'''


    def addChild(self, node):
        '''''添加一个子节点'''



class Tree(object):
        '''B树'''

        def __init__(self, size=6):
            self.size = size
            self.root = None
            self.length = 0

        def add(self, key, value=None):
            '''插入一条数据到B树'''


        def get(self, key):
            '''通过key查询一个数据'''

            node = self.__findNode(key)

            if node:
                return node.find(key).value

        def isEmpty(self):
            return self.length == 0

        def __findNode(self, key):
            '''通过key值查询一个数据在哪个节点,找到就返回该节点'''



        def delete(self, key):
            '''通过key删除一个数据项并返回它'''




        def __spilt(self, node):
            '''
            分裂一个节点，规则为:
            1、中间的数据项移到父节点
            2、新建一个右兄弟节点，将中间节点右边的数据项移到新节点
            '''





t = Tree(4)
t.add(20)
t.add(40)
t.add(60)
t.add(70, 'c')
t.add(80)
t.add(10)
t.add(30)
t.add(15, 'python')
t.add(75, 'java')
t.add(85)
t.add(90)
t.add(25)
t.add(35, 'c#')
t.add(50)
t.add(22, 'c++')
t.add(27)
t.add(32)

print (t.get(15))
