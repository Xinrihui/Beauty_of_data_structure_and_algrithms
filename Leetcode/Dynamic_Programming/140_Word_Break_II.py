#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np
class Solution:

    # wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
    def solve(self, s,wordDict):
        """


        时间复杂度 

        :param s: 
        :return: 
        """
        L_s=len(s)

        s=' '+s

        wordDict=set(wordDict)

        # dp=np.zeros((L_s+1),dtype=bool) # 方便调试

        dp=[False for __ in range(L_s+1)]  # 提高效率

        prev_dict=defaultdict(list)

        # 初始化
        dp[0]=True

        for i in range(1,L_s+1):

            flag=False
            for k in range(i):

                if dp[k]==True and s[k+1:i+1] in wordDict:
                    flag=True
                    prev_dict[i].append(k)

                    # break 要找出 所有的匹配

            if flag:
                dp[i]=True

        # print(prev_dict)

        # 4. 追踪解: 回溯得到解的集合
        self.break_list=[]

        current_idx=L_s
        prev=[]

        self.__process(prev_dict,s,current_idx,prev)

        # print(self.break_list)

        # 格式化 输出
        res=[]

        for i in range(len(self.break_list)-1,-1,-1):

            res.append(" ".join(self.break_list[i]))

        return res

    def __process(self,prev_dict,s,current_idx,prev):

        # 递归结束条件
        if current_idx==0:
            self.break_list.append(prev[::-1])

            return

        for idx in prev_dict[current_idx]:

            self.__process(prev_dict, s, idx, prev+[s[idx+1:current_idx+1]])







class Test:
    def test_small_dataset(self, func):

        s = "catsanddog"
        wordDict = ["cat", "cats", "and", "sand", "dog"]

        assert func(s,wordDict) == [
                                      "cats and dog",
                                      "cat sand dog"
                                    ]

        s = "pineapplepenapple"
        wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]

        assert func(s, wordDict) == [
                                      "pine apple pen apple",
                                      "pineapple pen apple",
                                      "pine applepen apple"
                                    ]

        s = "catsandog"
        wordDict = ["cats", "dog", "sand", "and", "cat"]

        assert func(s, wordDict) == []

        # TODO: 边界条件
        # assert func(None) == None
        #
        # assert func('') == ''

    def test_large_dataset(self, func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题

        Limits


        :param func: 
        :return: 
        """

        N = int(2 * pow(10, 4))
        max_v = int(pow(10, 9))

        l = np.random.randint(max_v, size=N)
        l1 = list(l)

        start = timeit.default_timer()
        print('run large dataset: ')
        func()
        end = timeit.default_timer()
        print('time: ', end - start, 's')


if __name__ == '__main__':

    sol = Solution()

    # IDE 测试 阶段：


    s = "catsanddog"
    wordDict = ["cat", "cats", "and", "sand", "dog"]

    # print(sol.solve(s,wordDict))

    s = "pineapplepenapple"
    wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]
    # print(sol.solve(s, wordDict))

    s = "catsandog"
    wordDict = ["cats", "dog", "sand", "and", "cat"]
    # print(sol.solve(s, wordDict))

    # IDE 测试 阶段：
    test = Test()
    test.test_small_dataset(sol.solve)

    # test.test_large_dataset(sol.solve)










