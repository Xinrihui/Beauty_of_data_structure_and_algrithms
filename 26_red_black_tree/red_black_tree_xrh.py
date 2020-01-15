#!/usr/bin/python
# -*- coding: UTF-8 -*-
from queue import Queue
import math

class TreeNode:

    def __init__(self, key=None,val=None, color=None):
        self.key=key
        self.val = val
        assert color in ['r', 'b']
        self.color = 'red' if color == 'r' else 'black'

        self.left = None
        self.right = None
        self.parent = None

    def is_black(self):

        return self.color == 'black'

    def is_red(self):
        return self.color == 'red'

    def set_black(self):
        self.color = 'black'
        return

    def set_red(self):
        self.color = 'red'


class RedBlackTree_recursive:
    """
    红黑树 递归 实现
    参考资料：
    1. 《算法(第四版)》
 
    """
    def __init__(self, key_list=None):
        self.root = None
        for key in key_list:
            self.put(key)

    def isRedNode(self,node):
        if node is None:
            return False
        return node.color == 'red'


    def rotateLeft(self,h):
        """
        把 朝右边的红链接 转换为 朝左边的红链接
        即 红色的右链接 -> 左链接
        :param h: 待旋转 子树的根节点
        :return x: 转换后的子树的 根节点
        """
        x = h.right
        h.right=x.left
        x.left=h

        x.color=h.color # 保持根节点的颜色
        h.set_red()

        return x

    def rotateRight(self,h):
        """
        红色的左链接 -> 右链接
        :param h: 
        :return: 
        """
        x = h.left
        h.left=x.right
        x.right=h

        x.color = h.color
        h.set_red()

        return x

    def flipcolor(self,h):
        """
        当子树的根节点的 左右链接 都是红链接时，要把红链接向上传递
        
        :param h: 子树的根节点
        :return: 
        """
        h.set_red()
        h.left.set_black()
        h.right.set_black()

    def put(self,key,val=0):

        self.root = self._put(self.root,key)
        self.root.set_black()

    def _put(self,p,key,val=0):

        if p is None:
            return TreeNode(key,val,'r') # 插入的新节点 必为红色

        if ord(key) == ord(p.key):
            p.val=val

        elif ord(key)>ord(p.key):
            p.right=self._put(p.right,key,val)
        elif ord(key)<ord(p.key):
            p.left=self._put(p.left,key,val)

        if self.isRedNode(p.right)==True and self.isRedNode(p.left)==False:
            p=self.rotateLeft(p)
        if self.isRedNode(p.left)==True and self.isRedNode(p.left.left)==True:
            p=self.rotateRight(p)
        if self.isRedNode(p.right)==True and self.isRedNode(p.left)==True:
            self.flipcolor(p)

        return p

    #  以下为 红黑树删除 所做的实现
    def moveflipColors(self,h):
        """
        用于删除节点的flipColor方法，该方法用于节点的合并，将父节点中的红色部分给与子节点
        :param h: 
        :return: 
        """
        h.set_black()
        h.left.set_red()
        h.right.set_red()

    def deleteMin(self):
        if self.isRedNode(self.root)==False and self.isRedNode(self.root)==False: #如果根节点的左右子节点是2-节点，我们可以将根设为红节点，
                                                                                   # 这样才能进行后面的moveRedLeft操作，因为左子树要从根节点借一个红色链接
            self.root.set_red()
        self.root=self._deleteMin(self.root)
        self.root.set_black() #借完以后，将根节点的颜色复原

    def _deleteMin(self,x):
        if x.left is None : return None # 没有比 x 更小的节点了 ，可以把x 删除

        if self.isRedNode(x.left) == False and self.isRedNode( x.left.left) == False: # 判断x的左节点是不是2-节点
            x=self.moveRedLeft(x)

        x.left=self._deleteMin(x.left)

        return self.balance(x) # 解除临时组成的4-节点


    def  moveRedLeft(self,h):
        """
        当前节点的左右子节点都是2-节点，左右子节点需要从父节点中借 两个红链接
        如果该节点的右节点的左节点是红色节点，说明兄弟节点不是2-节点，可以从兄弟节点中借一个
        :param h: 
        :return: 
        """
        self.moveflipColors(h) #从父节点中借一个

        if self.isRedNode(h.right.left): #判断兄弟节点，如果是非红节点，也从兄弟节点中借一个
            h.right=self.rotateRight(h.right)
            h=self.rotateLeft(h)
            self.flipcolor(h)  # 从兄弟节点借了一个红链接以后，我们就需要 将刚刚从父节点 借来的红链接 还给父节点了
        return h

    def balance(self,p):
        """
        
        :param h: 
        :return: 
        """
        if self.isRedNode(p.right):
            p=self.rotateLeft(p)

        if self.isRedNode(p.right)==True and self.isRedNode(p.left)==False:
            p=self.rotateLeft(p)
        if self.isRedNode(p.left)==True and self.isRedNode(p.left.left)==True:
            p=self.rotateRight(p)
        if self.isRedNode(p.right)==True and self.isRedNode(p.left)==True:
            self.flipcolor(p)

        return p

    def __repr__(self):

        return self._draw_tree()

    def _bfs(self):
        """
        bfs
        通过父子关系记录节点编号
        :return:
        """
        if self.root is None:
            return []

        ret = []
        q = Queue()
        # 队列[节点，编号]
        q.put((self.root, 1))

        while not q.empty():
            n = q.get()

            if n[0] is not None:
                ret.append((n[0].key, n[1]))
                q.put((n[0].left, n[1]*2))
                q.put((n[0].right, n[1]*2+1))

        return ret


    def _draw_tree(self):
        """
        可视化
        :return:
        """
        nodes = self._bfs()

        if not nodes:
            print('This tree has no nodes.')
            return

        layer_num = int(math.log(nodes[-1][1], 2)) + 1

        prt_nums = []

        for i in range(layer_num):
            prt_nums.append([None] * 2 ** i)

        for v, p in nodes:
            row = int(math.log(p, 2))
            col = p % 2 ** row
            prt_nums[row][col] = v

        prt_str = ''
        for l in prt_nums:
            prt_str += str(l)[1:-1] + '\n'

        return prt_str




if __name__ == '__main__':

    nums=['A','C','E','H','L','M','P','R','S','X','T']
    rbtree = RedBlackTree_recursive(nums)
    print(rbtree)

    rbtree.deleteMin()
    print(rbtree)


































