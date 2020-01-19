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

import bisect

class Node(object):
    '''''B树的节点'''

    def __init__(self):
        self.parent = None
        self.entitys = []
        self.childs = []

    def _find_insert_idx(self, l,v):
        """
        二分查找，找到有序数组的插入位置

        还可以直接使用 bisect 实现：
         http://kuanghy.github.io/2016/06/14/python-bisect

        :param v:
        :return:

    test.children=[Node('b') ,Node('d'),Node('e'),Node('f')]
    print(test._find_insert_idx( 'a'))  #[False, -1]
    print(test._find_insert_idx('c')) #[False,0]
    print(test._find_insert_idx('g')) #[False, 3]
    print(test._find_insert_idx('d')) #[True, 1, Node('d')]
        """
        left = 0
        right = len(l) - 1
        flag = False
        idx = 0

        while (left <= right):
            mid = ((left + right) // 2)  # Python 3以后  " / "就表示 浮点数除法，返回浮点结果;" // "表示整数除法。

            if (l[mid].key) == v:
                flag = True
                idx = mid
                break
            elif v > (l[mid].key):
                left = mid + 1  # mid+1 为了跳出循环
            elif v < (l[mid].key):
                right = mid - 1  # mid-1 为了跳出循环
        if left > right:
            idx = right
            return [flag, idx]

        return [flag, idx, l[idx]]

    def find(self, key):
        '''''通过key查找并返回一个数据实体 (下标,实体)  '''
        res=self._find_insert_idx(self.entitys,key)
        if res[0]==False:
            return  None
        else:
            return res[1:]


    def delete(self, key):
        ''''' 通过key删除一个数据实体,并返回它和它的下标(下标,实体) '''
        res = self._find_insert_idx(self.entitys, key)
        if res[0]==False: # 没有找到
            return  None
        else:
            del self.entitys[res[1]]
            return  res[1:]

    def isLeaf(self):
        '''''判断该节点是否是一个叶子节点'''
        return len(self.childs)==0


    def addEntity(self, entity):
        """
        添加一个数据实体
        并保持 entitys 的有序 
        :param entity: 
        :return: 
        """

        result = self._find_insert_idx(self.entitys, entity.key)

        if result[0] == False:

            idx=result[1]
            l=self.entitys

            if idx>=len(l)-1:
                l.append(entity)
            elif idx< 0:
                l.insert(0,entity)
            else:
                l.insert(idx+1,entity)

    def _find_insert_idx_childs(self, v):
        """
        对 self.childs 进行二分查找，找到有序数组的插入位置

        """
        l=self.childs
        left = 0
        right = len(l) - 1
        flag = False
        idx = 0

        while (left <= right):
            mid = ((left + right) // 2)  # Python 3以后  " / "就表示 浮点数除法，返回浮点结果;" // "表示整数除法。

            if (l[mid].entitys[-1] .key) == v:
                flag = True
                idx = mid
                break
            elif v > (l[mid].entitys[-1].key):
                left = mid + 1  # mid+1 为了跳出循环
            elif v < (l[mid].entitys[-1].key):
                right = mid - 1  # mid-1 为了跳出循环
        if left > right:
            idx = right
            return [flag, idx]

        return [flag, idx, l[idx]]


    def addChild(self, node):
        """
        添加一个子节点
        并保持 childs 的 有序性
        :param node: 
        :return: 
        """

        result=self._find_insert_idx_childs( node.entitys[-1].key ) #

        if result[0] == False:
            idx=result[1]
            l=self.childs

            if idx>=len(l)-1:
                l.append(node)
            elif idx< 0:
                l.insert(0,node)
            else:
                l.insert(idx+1,node)


class BTree(object):
        '''B树'''

        def __init__(self, key_list,size=2):
            self.size = size
            self.root = None
            self.length = 0

            for key,value in key_list:
                self.add(key,value)


        def add(self, key, value=None):
            """
            插入一条数据到B树
            若key 已存在，则更新 value 值
            :param key: 
            :param value: 
            :return: 
            """
            self.length+=1
            entity= Entity(key,value)

            if self.root:
                res=self.__findNode(key)
                Flag=res[0]
                current=res[1]
                if Flag: # key 已存在 则更新 value 值
                    res[2].value=value

                else: # key 不存在，则在叶子节点处进行插入
                    current.addEntity(entity)
                    if len(current.entitys) > self.size:
                        self.__spilt(current)

            else:# 插入第一个 节点
                self.root=Node()
                self.root.addEntity(entity)


        def get(self, key):
            '''通过key查询一个数据'''

            res = self.__findNode(key)
            Flag=res[0]
            node=res[1]

            if Flag:
                return node.find(key)[1].value

        def isEmpty(self):
            return self.length == 0

        def __findNode(self, key):
            """
            通过key值查询一个数据在哪个节点,找到就返回该节点
            :param key: 
            :return:  [Flag , node , entity] 
             其中  Flag ： 若找得到节点 为 True
                  node ： 若找得到，为 Key 对应的节点；若找不到则是 最后一次查找定位的叶子节点
                  entity：若找得到，为 Key 所在的节点 所 对应的 实体
            """

            if self.root:
                current = self.root
                while not current.isLeaf():

                    entitys= current.entitys
                    result = current._find_insert_idx(entitys,key)

                    if  result[0] : # 说明 找到节点了
                        return  [True,current,result[2]]

                    else: # result[0] == False # 说明没找到
                        idx = result[1]
                        l = current.entitys
                        if idx >= len(l) - 1: # key 比所有的entity 都大
                            current=current.childs[-1] # 去最右边的子树中找
                        elif idx < 0: # key 比所有的entity 都小
                            current = current.childs[0] # 去最左边的子树中找
                        else:
                            current= current.childs[ idx+1 ]

                # 如果已经 搜索 到了 叶子节点 ; 如果找不到会一直往下找直到遇到叶子节点
                entitys = current.entitys
                result = current._find_insert_idx(entitys, key)
                if  result[0] : # 说明 找到节点了
                    return  [True,current,result[2]]

                return [False,current,None]

        def delete(self, key):
            '''通过key删除一个数据项并返回它'''
            pass

        def __merge(self,node):
            pass

        def __spilt(self, node):
            '''
            分裂一个节点，规则为:
            1、中间的数据项移到父节点
            2、新建一个右兄弟节点，将中间节点右边的数据项移到新节点
            '''

            middle = len(node.entitys) // 2

            top = node.entitys[middle]

            right = Node()

            for e in node.entitys[middle + 1:]:
                right.addEntity(e)

            for n in node.childs[middle + 1:]:
                right.addChild(n)

            node.entitys = node.entitys[:middle]
            node.childs = node.childs[:middle + 1]

            parent = node.parent #

            if parent:
                parent.addEntity(top)
                parent.addChild(right)

                if len(parent.entitys) > self.size:
                    self.__spilt(parent)

            else: #没有父母
                self.root = Node() # 建立新的根节点
                self.root.addEntity(top)
                self.root.addChild(node)
                self.root.addChild(right)

                node.parent=self.root
                right.parent=self.root



if __name__ == '__main__':

    t = BTree([(30,'a'),(65,'b'),(99,'c'),(203,'d'),(105,'f')],2)
    # t.add(30,'a')
    print(t.get(105))
    # t.add(65)
    # t.add(99)
    # t.add(203, 'c')
    # t.add(80)
    # t.add(10)
    # t.add(30)
    # t.add(15, 'python')
    # t.add(75, 'java')
    # t.add(85)
    # t.add(90)
    # t.add(25)
    # t.add(35, 'c#')
    # t.add(50)
    # t.add(22, 'c++')
    # t.add(27)
    # t.add(32)
    #
    # print (t.get(15))
