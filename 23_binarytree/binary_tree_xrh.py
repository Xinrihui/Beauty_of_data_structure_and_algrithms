from collections import deque

class TreeNode(object):
    def __init__(self,item):
        self.val=item
        self.left=None
        self.right=None
        # self.height=None

class Solution1(object):
    """
    二叉树的链式存储法 表达二叉树
    """
    def buildTree(self, preorder,inorder):
        """
        用树的前序和中序遍历的结果来构建树
        :type preorder:  ['a','b','c','e','d']
        :type inorder:   ['c','b','e','a','d']
        :rtype: TreeNode
        """
        self.preorder = deque(preorder)
        self.inorder = deque(inorder)
        return self._buildTree(0, len(inorder))

    def _buildTree(self, start, end):
        if start<end:
            root_val=self.preorder.popleft()
            print("root: ",root_val )
            root=TreeNode(root_val)

            index=self.inorder.index(root_val,start,end) # 在数组的位置范围： [start,end) 中寻找 root_val
            root.left=self._buildTree(start,index)
            root.right=self._buildTree(index+1,end)

            return root

    def pre_order(self,root):
        if root is not None:
            print(root.val)
            self.pre_order(root.left)
            self.pre_order(root.right)

        return


    def in_order_depreatured(self,root):
        """
        非递归 实现树的中序遍历
        :param root: 
        :return: 
        """
        stack=[root]
        p=root
        res=[]

        while len(stack)!=0 :

            while (p!=None) and (p.left!=None) and (p.val not in res): #访问过的节点不要再入栈
                p = p.left
                stack.append(p)

            p=stack.pop()
            res.append(p.val)

            if p.right!=None:
                p=p.right
                stack.append(p)

        return res

    def in_order(self, root):
        """
        非递归 实现树的中序遍历
        :param root: 
        :return: 
        """
        stack = []
        p = root
        res = []

        while p!=None or len(stack)!=0:

            if p!=None: # p 不为空就入栈
                stack.append(p)
                p=p.left #指向左节点

            else: # 如果p 为空就弹出
                p=stack.pop() # 访问中间节点
                res.append(p.val)
                p=p.right  # 指针指向右子树

        return res

    def _depth_recursion(self,root):
        if root is None:
            return 0
        left_depth= self._depth_recursion(root.left)
        right_depth=self._depth_recursion(root.right)

        return max(left_depth,right_depth)+1


    def _depth(self, root):
        """
        改进层次遍历 ，把树的各个层都切分出来，并能输出树的高度
        :type root: TreeNode
        :rtype: int
        """
        Queue = deque()
        Queue.append(root)
        depth = 0
        while (len(Queue) != 0):
            depth += 1
            n = len(Queue)
            for i in range(n):  # Stratified according to depth

                target = Queue.popleft()
                print(target.val)
                print('depth: ', depth)

                if target.left != None:
                    Queue.append(target.left)
                if target.right != None:
                    Queue.append(target.right)

        return depth

class Solution2(object):
    """
    基于数组的顺序存储法 表达二叉树
    """
    def pre_order(self, tree_array):
        """
        前序遍历 中->左->右
        :param tree_array: 
        :return: 
        """
        stack=[]
        i=1
        node=[tree_array[i],i]
        stack.append(node)
        result=[]
        while ( len(stack)!=0 ):
            current=stack.pop()
            # print(current)
            result.append(current[0])
            i=current[1]

            if 2*i+1<len(tree_array) and tree_array[2*i+1]!=None: # tree_array 越界 访问检查 : 2*i+1<len(tree_array)
                node=[tree_array[2*i+1],2*i+1]
                stack.append(node)
            if  2*i<len(tree_array) and tree_array[2*i]!=None:
                node = [tree_array[2 * i ], 2 * i]
                stack.append(node)
        return result

    def post_order(self, tree_array):
        """
        前序遍历 ：中->左->右
        前序遍历反过来 ：中->右->左
        前序遍历反过来再逆序 ： 左 -> 右 ->中 （后序遍历）
        
        https://www.cnblogs.com/bjwu/p/9284534.html  
        :param tree_array: 
        :return: 
        """
        stack=[]
        i=1
        node=[tree_array[i],i]
        stack.append(node)
        result=[]
        while ( len(stack)!=0 ):
            current=stack.pop()
            # print(current)
            result.append(current[0])
            i=current[1]

            if  2*i<len(tree_array) and tree_array[2*i]!=None:
                node = [tree_array[2 * i ], 2 * i]
                stack.append(node)

            if 2*i+1<len(tree_array) and tree_array[2*i+1]!=None: # tree_array 越界 访问检查 : 2*i+1<len(tree_array)
                node=[tree_array[2*i+1],2*i+1]
                stack.append(node)

        return result[::-1]  # 逆序输出即为 后序遍历

    def in_order_deprecated(self, tree_array):
        stack=[]
        i=1
        result=[]
        while ( i < len(tree_array) and tree_array[i] != None)  or (len(stack) != 0):  #   ( i < len(tree_array) and tree_array[i] != None) 等价于 p != None
            while (i < len(tree_array) and tree_array[i] != None):
                node = [tree_array[i],  i]
                stack.append(node)
                i = 2 * i  # 左子树全部进栈

            if (len(stack) != 0) :
                current = stack.pop()  #
                # print(current)
                result.append(current[0])
                i = current[1]
                i= 2*i+1 #尝试去访问右子树

        return result

    def in_order(self, tree_array):
        """
        好理解 
        :param tree_array: 
        :return: 
        """
        stack=[]
        i=1
        result=[]
        while ( i < len(tree_array) and tree_array[i] != None)  or (len(stack) != 0):
            if (i < len(tree_array) and tree_array[i] != None):
                node = [tree_array[i],  i]
                stack.append(node)
                i = 2 * i  # 左子树全部进栈
            else:
                current = stack.pop()  #
                # print(current)
                result.append(current[0])
                i = current[1]
                i= 2*i+1 #尝试去访问右子树

        return result

    def hierarchy_order(self, tree_array):
        """
        树的层次遍历 （广度优先遍历）
        :param tree_array: 
        :return: 
        """
        fifo=deque()
        i=1
        node=[tree_array[i],i]
        fifo.appendleft(node)

        result=[]

        while ( len(fifo)!=0 ):
            current=fifo.pop()
            # print(current)
            result.append(current[0])
            i=current[1]

            if  2*i<len(tree_array) and tree_array[2*i]!=None: # 左边
                node = [tree_array[2 * i ], 2 * i]
                fifo.appendleft(node)

            if 2*i+1<len(tree_array) and tree_array[2*i+1]!=None: # 右边
                node=[tree_array[2*i+1],2*i+1]
                fifo.appendleft(node)

        return result


if __name__ == "__main__":
    #solution1
    # preorder=['a','b','c','e','d']
    # inorder= ['c','b','e','a','d']

    preorder=['A','B','D','F','G','C','E','H']
    inorder=['F','D','G','B','A','E','H','C']
    postorder= ['F','G','D','B','H','E','C','A']

    solution=Solution1()
    root=solution.buildTree(preorder,inorder)
    # solution.pre_order(root)
    print(solution.in_order(root))
    # print(solution._depth(root))
    # print(solution._depth_recursion(root))

    #solution2
    # tree_array=[None,'A','B','C','D',None,'E',None,'F','G',None,None,None,'H']
    # solution2 = Solution2()
    # print('preorder: ',solution2.pre_order(tree_array))
    # print('inorder: ',solution2.in_order(tree_array))
    # print('postorder: ', solution2.post_order(tree_array))
    # print('hierarchy_order: ', solution2.hierarchy_order(tree_array))


