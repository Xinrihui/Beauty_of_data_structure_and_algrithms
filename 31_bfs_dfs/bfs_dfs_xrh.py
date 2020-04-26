#!/usr/bin/python
# -*- coding: UTF-8 -*-

from collections import *

class solution():



    def get_path_v1(self,prev,s,t):
        """
        prev :用来记录搜索路径。当我们从顶点 s 开始，广度优先搜索到顶点 t 后，prev 数组中存储的就是搜索的路径。不过，这个路径是反向存储的。prev[w]存储的是，顶点 w 是从哪个前驱顶点遍历过来的。比如，我们通过顶点 2 的邻接表访问到顶点 3，那 prev[3]就等于 2
        递归访问  prev 得到 路径
        :param prev: 前置数组 
        :param s: 起点
        :param t: 终点
        :return: 
        """

        path=[]

        current=t

        while current!=-1:

            path.append(current)

            if current==s:
                break

            current=prev[current]

        else:
            return []

        return path[::-1]

    def get_path_v1_1 (self,prev,s,t):
        """
        在 get_path_v1 基础上， 用栈实现 path[::-1]
        
        :param prev: 
        :param s: 0
        :param t: 6
        :return: 
        """

        stack=[]
        current=t

        path=[]

        while current!=-1:

            stack.append(current)

            if current==s:
                break

            current=prev[current]

        while len(stack)>0: # stack=[6,4,3,0]

            path.append(stack.pop())

        return path


    def bfs(self,graph,s,t):
        """
        广度优先搜索图
        （1）在无向图中 从 起点s 开始 搜索到 终点t，并记录搜索路径
        （2）通过前置数组 prev 记录路径，数组中的 元素 对应每一个节点的 前驱节点
         
         ref: https://time.geekbang.org/column/article/70891
         
        :param graph:    
        :param s:  
        :param t: 
        :return: 
        """
        queue=deque()
        queue.append(s)

        visited=set()
        visited.add(s) # 初始化，起点 s 需标记为 已被访问

        prev=[-1]*len(graph)

        while len(queue)>0:

            N=len(queue)

            for i in range(N):

                current=queue.popleft()
                print(current) # 访问节点

                for node in graph[current]: # 遍历 current节点 周围的子节点

                    if node == t:
                        prev[node]=current  # 记录 node 的 父亲节点为 current

                        return True,self.get_path_v1(prev,s,t)

                    if node not in visited: # 已经访问过的节点 无需再加入 queue
                        queue.append(node)
                        visited.add(node) # 加入 queue 就代表 被访问，避免 由于环路的存在而 导致 queue 存在重复元素；
                                         # eg. 在访问1节点时，会把4节点加入队列，访问3节点时，也会把4节点加入 队列，这样队列中就存在着重复的 4 节点
                        prev[node] = current

        return False,[]

    def find_xD_friends(self,graph,s,X=2):
        """
        给一个用户，如何找出这个用户的所有 X 度（其中包含一度、二度...X度）好友关系
        
        1.使用 广度优先搜索
        
        :param graph: 
        :param s: 
        :return: 
        """

        queue=deque()
        queue.append(s)

        visited=set()
        visited.add(s)  # 初始化，起点 s 需标记为 已被访问

        level=0 # 度
        all_level_friends=[] # 所有度的 好友

        while len(queue)>0:

            N=len(queue)

            level_friends=[]

            for i in range(N):

                current=queue.popleft()
                print(current) # 访问节点

                level_friends.append(current) # 加入 X度好友 集合

                for node in graph[current]: # 遍历 current节点 周围的子节点

                    if node not in visited: # 已经访问过的节点 无需再加入 queue
                        queue.append(node)
                        visited.add(node)


            print('level:',level,' friends:',level_friends)
            all_level_friends.append(level_friends)

            if level == X:
                break

            level+=1


        return all_level_friends[1:]



    def dfs(self,graph,s,t):

        """
        深度优先搜索
        起点为 s ，终点为 t 
        
        1.用栈 实现 (非递归)
        2.返回搜索的路径，并不是 s 到 t 的最短路径
        
        :param graph: 
        :param s: 
        :param t: 
        :return: 
        """

        stack=[]
        stack.append(s)
        visited=set()
        visited.add(s) # 初始化，起点 s 需标记为 已被访问

        prev = [-1] * len(graph)

        while len(stack) > 0:

            current=stack.pop()
            print(current)  # 访问节点

            for node in graph[current]:  # 遍历 current节点 周围的子节点

                if node == t:
                    prev[node] = current  # 记录 node 的 父亲节点为 current

                    return True, self.get_path_v1(prev, s, t)

                if node not in visited:  # 已经访问过的节点 无需再加入 queue
                    stack.append(node)
                    visited.add(node) # 加入 stack 就算被访问过

                    prev[node] = current

        return False,[]


    def dfs_v1(self,graph,s,t):
        """
        深度优先搜索
        起点为 s ，终点为 t 

        1.递归实现
        2.返回搜索的路径

        :param graph: 
        :param s: 
        :param t: 
        :return: 
        """
        self.graph=graph
        self.end=t

        p=s
        prev_path=[p]

        self.visited = set()
        self.visited.add(s)

        path=self._dfs_recursive(p,prev_path)

        return path

    def _dfs_recursive(self,current,prev_path):

        # print(prev_path)

        if current==self.end: # 递归结束条件
            return prev_path

        for node in self.graph[current]:  # 遍历 current节点 周围的子节点

            if node not in self.visited:  # 已经访问过的节点 无需再加入 queue

                self.visited.add(node)  # 加入 stack 就算被访问过

                path=self._dfs_recursive(node,prev_path+[node])

                if path !=None:
                    return path

    def find_xD_friends_v1(self,graph,s,X=2):
        """
        给一个用户，如何找出这个用户的所有 X 度（其中包含一度、二度...X度）好友关系
        
        1.使用 深度优先搜索
        
        :param graph: 
        :param s: 
        :return: 
        """
        self.graph=graph
        self.X=X

        p=s
        level=0

        self.visited = set()
        self.visited.add(s)

        self.all_level_friends ={}  # 所有度的 好友

        for i in range(1,self.X+1):
            self.all_level_friends[i]=set() #初始化 all_level_friends

        self._dfs_recursive_v1(p,level)

        return self.all_level_friends

    def _dfs_recursive_v1(self,current,level):


        if level==self.X+1: # 递归结束条件
            return

        if level!=0:
            self.all_level_friends[level].add(current)

        for node in self.graph[current]:  # 遍历 current节点 周围的子节点

            if node not in self.visited:  # 已经访问过的节点 无需再加入 queue

                self.visited.add(node)  # 加入 stack 就算被访问过

                self._dfs_recursive_v1(node,level+1)




if __name__ == '__main__':

    sol=solution()

    G = {
        0: set([1, 3]),
        1: set([0,2,4]),
        2: set([1,5]),
        3: set([0,4]),
        4: set([1,3,5,6]),
        5: set([2,4,7]),
        6: set([4,7]),
        7: set([5,6])
        }

    # print(sol.bfs(G,0,6))

    # print(sol.bfs(G, 0, 2))

    # print(sol.dfs(G, 0, 2))
    # print(sol.dfs_v1(G, 0, 2))

    # print(sol.dfs_v1(G, 0, 3))

    # print(sol.find_xD_friends(G,0,2))

    # print(sol.find_xD_friends_v1(G,0,2))












