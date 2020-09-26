#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import *

from collections import *

import timeit


class Test:

    def building_main_string(self,s1,s2,N,A,B,C,D):


        x1, x2 = ord(s1), ord(s2)

        x_list = [0] * N
        x_list[0], x_list[1] = x1, x2

        s_list = [''] * N
        s_list[0], s_list[1] = s1, s2

        for i in range(2, N):
            x_list[i] = (A * x_list[i - 1] + B * x_list[i - 2] + C) % D

        for i in range(2, N):
            s_list[i] = chr(97 + (x_list[i] % 26))

        return s_list

    def test_small_dataset(self,func):

        s1, s2, N, A, B, C, D ='a','a',50,1,1,1,30

        dictionary=['axpaj', 'apxaj', 'dnrbt', 'pjxdn', 'abd']
        main_string=self.building_main_string(s1, s2, N, A, B, C, D)


        assert  func(dictionary, main_string)==4


        # 边界情况 的测试 (corner case) :

        # s1, s2, N, A, B, C, D ='a','a',2,1,1,1,30
        # dictionary=['axpaj', 'apxaj', 'dnrbt', 'pjxdn', 'abd']
        # main_string=self.building_main_string(s1, s2, N, A, B, C, D)
        # print(main_string)
        # assert  func(dictionary, main_string)==4

        # assert  func()==''
        # assert  func()==''


    def test_large_dataset(self,func):
        """
        自己 生成大的 数据集，查看算法效率，解决 TTL 问题
        
        Limits
        
        Time limit: 150 seconds.
        
        1 ≤ L (字典中单词个数) ≤ 2*10^4.
        
        词典中没有两个单词是相同的 
        词典中的每个单词的长度在 2 - 10^5个字母（含）之间
        词典中所有单词的长度总和不超过 10^5
        
        2 ≤ N (主串的长度) ≤ 10^6.
        
        0 ≤ A ≤ 10^9.
        0 ≤ B ≤ 10^9.
        0 ≤ C ≤ 10^9.
        1 ≤ D ≤ 10^9.
        
        :param func: 
        :return: 
        """

        # 1. 生成 随机的 大词典

        dictionary = []

        L = int(2e4) # 词典中单词的个数 20000 个

        for i in range(L):

            word_length=random.randint(2, 11, size=1) # 单词的长度 [2,11)

            ascii_char_list = random.randint(97, 97 + 26, size=word_length) # 只能由 26个 小写字母组成 单词

            word = ''.join(list(map(lambda c: chr(c), ascii_char_list)))

            dictionary.append(word)


        # 2.生成 大主串

        s1, s2= 'a', 'a'

        N= int(1e6) # 主串的长度

        A= random.randint(0, int(1e9)+1, size=1)
        B = random.randint(0, int(1e9)+1, size=1)
        C = random.randint(0, int(1e9)+1, size=1)
        D = random.randint(1, int(1e9)+1, size=1)

        main_string = self.building_main_string(s1, s2, N, A, B, C, D)

        start = timeit.default_timer()
        print('run large dataset: ')
        func(dictionary, main_string)
        end = timeit.default_timer()
        print('time: ', end - start, 's')



class solutions:
    """
   
    """

    def _hash_string_next(self, s , pre_counter=None ,sub_char=None,add_char=None ):
        """
        自定义的 hash 函数：在 def _hash_string() 基础上 ，增加 对迭代计算的支持

        字符串 s (由 26个小写 英文字符 组成) ,长度为 n 

        1. 利用 Counter (速度更快) 统计 s[1:n-1] 中 每一个字符 出现的次数，并转换为 词袋向量 

        s='aaaba'
        s[1:4]='aab'
                      a b c     z
        s_mid_vector=[2,1,0,...,0]

        2. s_mid_vector 加上 首尾 字符的信息 

        s_vector=[ord('a'),ord('a')]+[2,1,0,...,0] 

        3. 计算s_vector 的 hash 值 

        :param s: 
        :param pre_counter: 前一次  滑动窗口统计 的各个字符的频率
        :param sub_char: 当前的窗口 与前一次的窗口比 少了哪个字符
        :param add_char: 当前的窗口 与前一次的窗口比 多了哪个字符
        :return: 
        """

        n=len(s)

        base_ASCII = 97  # 'a'的 ASCII 为 97

        if pre_counter==None:

            #1. 统计 s[1:n-1] 中 每一个字符 出现的次数
            s_mid_freq = Counter(s[1:n-1])

        else: # 迭代计算：用前一次 滑动窗口统计各个字符的频率，加上本次滑动窗口 的变化的字符的频率，即可得到本次 滑动窗口 中各个字符的频率
            pre_counter[sub_char]-=1

            pre_counter[add_char]+=1

            s_mid_freq= pre_counter

        # 将 s_mid 中 26个字符的 频率转换为 向量
        s_mid_vector = [0] * 26

        for ele in s_mid_freq.items():

            s_mid_vector[ord(ele[0])-base_ASCII]=ele[1]

        #2. s_mid_vector 加上 首尾 字符的信息
        s_vector = [ord(s[0]), ord(s[-1])] + s_mid_vector

        #3. s_vector 计算 hash 值
        return s_mid_freq,hash(tuple(s_vector))



    def _hash_string(self, s):

        """
        自定义的 hash 函数：
        
        字符串 s (由 26个小写字母 组成) ,长度为 n 
        
        1.统计 s[1:n-1] 中 每一个字符 出现的次数，生成 词袋向量 
        
        s='aaaba'
        s[1:4]='aab'
                      a b c     z
        s_mid_vector=[2,1,0,...,0]
         
        2. s_mid_vector 加上 首尾 字符的信息 
         
        s_vector=[ord('a'),ord('a')]+[2,1,0,...,0] 
        
        3. s_vector 计算 hash 值 
        
        :param s: 
        :return: 
        """
        n=len(s)

        s_mid_vector = [0] * 26

        base_ASCII = 97  # 'a'的 ASCII 为 97

        for c in s[1:n-1]:
            s_mid_vector[ord(c) - base_ASCII] += 1

        s_vector = [ord(s[0]), ord(s[-1])] + s_mid_vector

        return hash(tuple(s_vector))


    def _string_to_hash(self, s):
        """
        记录 字符串中 每一个 字符出现的频率，并将 词袋模型 编码的向量 转换为 hash 值；

        这样 比较两个 字符串 是否相等 直接比较 Hash 值即可

        eg.

        s='aab' 和  s='aba'      

        a 出现 2 次，b 出现 1 次 
        s_vector=(2,1,0,...,0)

        s_hash= my_hash(s_vector)

        :param s: 
        :return: 
        """

        s_vector = [0] * 26

        base_ASCII = 97  # 'a'的 ASCII 为 97

        for c in s:
            s_vector[ord(c) - base_ASCII] += 1

        # TODO: 自定义 Hash 函数
        # s_hash = 0
        # seed=26
        # for index, time in enumerate(s_vector):
        #
        #     s_hash= s_hash*seed +time

        return hash(tuple(s_vector))


    def _string_to_hash_Deprecated(self,s):
        """
        记录 字符串中 每一个 字符出现的频率，并将 词袋模型 编码的向量 转换为 hash 值；

        这样 比较两个 字符串 是否相等 直接比较 Hash 值即可

        eg.

        s='aab' 和  s='aba'

        s_vector=(2,1,0,...,0)

        a 出现 2 次，b 出现 1 次  => '2a1b'

        s_hash= hash('2a1b')

        :param s:
        :return:
        """
        # import murmurhash
        s_vector=[0]*26

        base_ASCII=97  #'a'的 ASCII 为 97

        for c in s:

            s_vector[ord(c)-base_ASCII]+=1

        s_hash=[]

        for index,time in enumerate(s_vector):

            if time!=0:
                s_hash.append( str(time)+chr(index+base_ASCII))

        # s_hash = murmurhash.hash(''.join(s_hash)) #TODO： 没有 murmurhash 包导致 提交后 报错 RE (runtime error)

        s_hash= hash(''.join(s_hash))

        return s_hash



    def _string_to_vector(self, s):
        """
        将 字符串 转换为 词袋模型（词袋： ASCII 中小写的 26个字母）编码的向量，
        即 统计 字符串中 每一个 字符 出现的次数（频率）
        
        eg. 
        s='aab'
                  a b c     z
        s_vector=(2,1,0,...,0)
        
        s='aba'
        s_vector=(2,1,0,...,0)
        
        s='ac'
        s_vector=(1,0,1,...,0)
        
        :param s: 
        :return: 
        """

        s_vector=[0]*26

        base_ASCII=97  #'a'的 ASCII 为 97

        for c in s:

            s_vector[ord(c)-base_ASCII]+=1

        return tuple(s_vector)


    def _match_one_pattern(self,pattern,main_string):
        """
        
        :param pattern: 'axpaj'
        :param main_string: 'aapxjdnrbtvldptfzbbdbbzxtndrvjblnzjfpvhdhhpxjdnrbt'
        :return: 
        """
        pattern=pattern.strip() # 去掉 模式串 首尾的空格

        main_string_length=len(main_string)
        pattern_length=len(pattern)

        pattern_mid_vector=self._string_to_vector(pattern[1:pattern_length-1]) # 预先计算：将 pattern 中间的 字符串 向量化

        i=0

        while i <= main_string_length-pattern_length: # 滑动窗口

            sub_main_string =main_string[i:i+pattern_length] # 主串在 窗口中的子串

            if sub_main_string[0]==pattern[0] and sub_main_string[-1]==pattern[-1]: # 首尾 字母是否相同

                # 不能用 set 比较，因为 set 会对所有的 char 去重：set('bdd')== set('bbd')
                # if pattern_mid_set== set(sub_main_string[1:pattern_length-1]):
                #     return True,i

                if pattern_mid_vector== self._string_to_vector(sub_main_string[1:pattern_length-1]):
                    return True,i

            i+=1

        return False,0

    def scrambled_words_naive(self, dictionary, main_string):
        """
        暴力解法 

        :param dictionary: 模式串集合 ['axpaj', 'apxaj', 'dnrbt', 'pjxdn', 'abd']
        :param main_string: 主串 'aapxjdnrbtvldptfzbbdbbzxtndrvjblnzjfpvhdhhpxjdnrbt'
        :return: 
        """

        time=0

        for pattern in dictionary:

            flag , _ =self._match_one_pattern(pattern,main_string)

            if flag==True:
                time+=1

        return time


    def scrambled_words(self, dictionary, main_string):
        """
        对 naive 的方法 进行改进
        
        大数据集的规模为：
        
        Time limit: 150 seconds.
        1 ≤ L (字典中单词个数) ≤ 20000.
        2 ≤ N (主串的长度) ≤ 10^6.
        
        分析 TTL 的 原因为 :
        
        1.主串的 长度 很长，每一次 滑动 窗口( 窗口 大小与 模式串 长度相同 ) 与 模式串 进行匹配 都要 计算 窗口中主串的子串的向量；
        2. 字典中 的 模式串个数 很多，对于每一个 模式串 都要 去滑动 主串的窗口 
         
        解决步骤如下：
        
        step1. 统计 所有 模式串的 长度，把长度相同的 放到一起
        
        step2. 对于 特定长度的 模式串 的集合 进行处理
            
        
        :param dictionary:  ['axpaj', 'apxaj', 'dnrbt', 'pjxdn', 'abd']
        :param main_string: 'aapxjdnrbtvldptfzbbdbbzxtndrvjblnzjfpvhdhhpxjdnrbt'
        :return: 
        """

        #step1.统计 所有 模式串的 长度，把长度相同的 放到一起
        dic_word_length={}

        for pattern in dictionary:

            pattern = pattern.strip()  # 去掉 模式串 首尾的空格

            if len(pattern) not in dic_word_length:

                dic_word_length[len(pattern)]=[pattern]

            else:
                dic_word_length[len(pattern)].append(pattern)

        # dic_word_length {5: ['axpaj', 'apxaj', 'dnrbt', 'pjxdn'], 3: ['abd']}

        total_time=0 # 总共的 匹配次数

        for ele in dic_word_length.items(): # ele=( 5 , ['axpaj', 'apxaj', 'dnrbt', 'pjxdn'])

            length=ele[0] # 模式串 的长度
            pattern_list=ele[1]

            # step2. 对于 特定长度的模式串进行处理：
            time=self._match_equal_length_pattern_byHash(length,pattern_list,main_string)

            total_time+=time

        return total_time

    def _match_equal_length_pattern_byHash(self, pattern_length, pattern_list, main_string):
        """
        对于 特定长度的 模式串 集合 进行处理：
        
        1.使用自定义的 hash 函数 计算每一个模式串 的 hash 值，放入 hashtable 中，hashtable 的结构为 { hashcode: pattern_small_list } （由于 pattern 的 hashcode 可能会相同,相同的 都放入一个 pattern_small_list 中）
        
        2. 滑动 主串的 滑动窗口， (采用 迭代计算 的方法) 计算 滑动窗口中的 hashcode ，判断 hashcode是否在 hashtable 中存在:
            若存在，把 hashcode 对应的 pattern_small_list 中的 每一个  pattern 都加入 匹配集合 set() 中 ( set() 自动去重) ;
            
            对比 def _match_equal_length_pattern_byHash_Deprecated ，对于 pattern_list 中所有的 pattern 我只用 滑动一次 主串即可，降低了时间复杂度
            
            
        3. 匹配集合 的长度  即为 pattern_list 中 与 主串匹配上的 pattern 的个数
        
        
        pattern_list 中 pattern的个数 为 l 
        main_string 的长度为 N 
        
        算法的时间复杂度： O( l + N ) 
        空间复杂度 ： O(l)
        
        -----------------------
        
        自定义的 hash 函数：
        
        字符串 s (由 26个小写字母 组成) ,长度为 n 
        
        1.统计 s[1:n-1] 中 每一个字符 出现的次数，生成 词袋向量 
        
        s='aaaba'
        s[1:4]='aab'
                      a b c     z
        s_mid_vector=[2,1,0,...,0]
         
        2. s_mid_vector 加上 首尾 字符的信息 
         
        s_vector=[ord('a'),ord('a')]+[2,1,0,...,0] 
        
        3. 计算 s_vector 的 hash 值 
        
        ----------------------------
        
        迭代计算（增量计算） 的方法 拿到 滑动窗口中子串的 hashcode ：
        
        1.每一次 向前(向右)滑动 窗口 一个 位置, 统计 窗口 中字符的 频率可以 使用 上一次的 窗口中 字符的频率信息 + 当前的 增量信息： 
          
          eg. 
          main_string = 'aapxjdnrbt'
          
          第 1个窗口 [aapxj]dnrbt  s='aapxj' 小窗口 s[1:4]='apx' s_mid_vector_1= 
          
          第 2个窗口 a[apxjd]nrbt  s='apxjd' 小窗口 s[1:4]='pxj' 
          
          s_mid_vector_1  中 'a'的频率 -1  'j'的频率 +1 =>
          s_mid_vector_2 =
          
        
        :param pattern_length: 5
        :param pattern_list: ['axpaj', 'apxaj', 'dnrbt', 'pjxdn']
        :param main_string: 'aapxjdnrbtvldptfzbbdbbzxtndrvjblnzjfpvhdhhpxjdnrbt'
        :return: 匹配的 模式串的个数 
        """

        # 1.使用自定义的 hash 函数 计算每一个模式串 的 hash 值，放入 hashtable 中，hashtable 的结构为 { hashcode: pattern_small_list }
        # （由于 pattern 的 hashcode 可能会相同,相同的 都放入一个 pattern_small_list 中）

        pattern_hashtable={}

        print('pattern_length:{} , num: {} , pattern_list[0]:{} '.format(pattern_length,len(pattern_list),pattern_list[0]))

        start = timeit.default_timer()

        for pattern in pattern_list:


            _,pattern_hashcode=self._hash_string_next(pattern)

            if pattern_hashcode in pattern_hashtable:

                pattern_hashtable[pattern_hashcode].append(pattern)

            else:

                pattern_hashtable[pattern_hashcode]=[pattern]


        # 2. 滑动 主串的 滑动窗口， 采用 迭代计算 的方法 拿到 滑动窗口中的 hashcode ，判断 hashcode是否在 hashtable 中存在
        #    若存在，把 hashcode 对应的 pattern_small_list 中的 每一个  pattern 都加入 匹配集合 set() 中

        end = timeit.default_timer()
        print('all pattern to hashcode cost time: {}s'.format(end - start))

        main_string_length = len(main_string)

        match_set=set() # 匹配集合

        print('match main_string by Sliding window ... ')

        start = timeit.default_timer()

        i = 0
        counter, sub_main_string_hashcode = self._hash_string_next(main_string[i:i + pattern_length])
        if sub_main_string_hashcode in pattern_hashtable:
            match_set = match_set | set(pattern_hashtable[sub_main_string_hashcode])  # 集合 求 并集
        i += 1

        while i <= (main_string_length - pattern_length):  # 滑动窗口 TODO: 时间开销 的最大头，造成 TTL的 首要罪人

            # 迭代计算 , 不用每次 都 统计 main_string[i:i + pattern_length] 中各个 字符 出现频率
            counter,sub_main_string_hashcode = self._hash_string_next(main_string[i:i + pattern_length],counter,
                                                                        sub_char=main_string[i],add_char=main_string[i +pattern_length-2])

            if sub_main_string_hashcode in pattern_hashtable: # 若窗口中的子串的 hashcode 与 模式串的hashcode 匹配，把 该 hashcode 对应的 pattern_small_list 中的 每一个  pattern 都加入 匹配集合

                match_set= match_set | set(pattern_hashtable[sub_main_string_hashcode]) # 集合 求 并集


            i += 1

        #3. 匹配集合 的长度  即为 pattern_list 中 与 主串匹配上的 pattern 的个数
        time=len(match_set)

        end = timeit.default_timer()
        print(' Sliding window cost time: {}s'.format(end - start))

        return time

    def _match_equal_length_pattern_byHash_Deprecated(self,pattern_length,pattern_list,main_string):

        """
        
        遍历每一个 模式串：
        模式串 的 hash 值 与  主串的 滑动窗口 中的 子串的 hash 值进行比较，若相等 说明 匹配上，计数器 +1 
        
        pattern_list 中 pattern的个数 为 l 
        main_string 的长度为 N 
        
        算法的时间复杂度： O( l*N )  大数据集 还是 TTL
        
        :param pattern_length: 5
        :param pattern_list: ['axpaj', 'apxaj', 'dnrbt', 'pjxdn']
        :param main_string: 'aapxjdnrbtvldptfzbbdbbzxtndrvjblnzjfpvhdhhpxjdnrbt'
        :return: 匹配的 模式串的个数 
        """

        #2.1 由于 模式串的 长度 决定了 窗口的大小，所以 预先 计算 在当前 窗口大小下，主串的 所有 窗口中的子串 的 hashcode，并缓存

        main_string_hashcode_cache=[]

        i = 0

        main_string_length=len(main_string)

        while i <= ( main_string_length - pattern_length):  # 滑动窗口

            sub_main_string = main_string[i:i + pattern_length]  # 主串在 窗口中的子串

            sub_main_string_hashcode=self._string_to_hash(sub_main_string[1:pattern_length-1])

            main_string_hashcode_cache.append(sub_main_string_hashcode)

            i+=1

        # 2.2 取其中一个 模式串，计算 模式串的 hashcode ，在主串 滑动窗口 直接进行匹配 （窗口中的子串的 hashcode 已经计算好了）

        time=0

        for pattern in pattern_list:

            print(pattern) # TODO: 提交 记得 注释

            pattern_mid_hashcode = self._string_to_hash( pattern[1:pattern_length - 1] )  # 预先计算：将 pattern 中间（去除首尾字符）的 字符串 计算hash 值

            i = 0

            while i <= (main_string_length - pattern_length):  # 滑动窗口 TODO： 对于每一个 pattern 都要 来滑动窗口，而  pattern_list 中的 pattern 个数有 1e4 ，因此 此 循环内的 代码效率为优化 重点

                # sub_main_string = main_string[i:i + pattern_length]  # 主串在 窗口中的子串 ; TODO: List 的切片 实际为复制，有时间开销

                if main_string[i:i + pattern_length][0] == pattern[0] and main_string[i:i + pattern_length][-1] == pattern[-1] and pattern_mid_hashcode == main_string_hashcode_cache[i]:  # 判断 首尾 字母 都相同 并且 hash 值相等

                        time+=1

                        break # 匹配 一个 即可退出

                i += 1

        return time

    def _match_equal_length_pattern_byVector(self, pattern_length, pattern_list, main_string):
        """
        
        2.1  由于 模式串的 长度 决定了 窗口的大小，所以 预先 计算 在当前 窗口大小下，主串的 所有 窗口中的子串 的向量，并缓存 
            
        2.2  取其中一个 模式串，计算 模式串的 向量，滑动 窗口 与 主串 直接 进行 匹配 （窗口中的子串的向量已经计算好了） 
        
        :param pattern_length: 5
        :param pattern_list: ['axpaj', 'apxaj', 'dnrbt', 'pjxdn']
        :param main_string: 'aapxjdnrbtvldptfzbbdbbzxtndrvjblnzjfpvhdhhpxjdnrbt'
        :return: 匹配的 模式串的个数 
        """

        # 2.1 由于 模式串的 长度 决定了 窗口的大小，所以 预先 计算 在当前 窗口大小下，主串的 所有 窗口中的子串 的向量，并缓存
        main_string_vector_cache = []

        i = 0

        while i <= (len(main_string) - pattern_length):  # 滑动窗口

            sub_main_string = main_string[i:i + pattern_length]  # 主串在 窗口中的子串

            sub_main_string_vector=self._string_to_vector(sub_main_string[1:pattern_length-1]) # 子串 向量化


            main_string_vector_cache.append(sub_main_string_vector)

            i += 1

        # 2.2 取其中一个 模式串，计算 模式串的 向量，滑动 窗口 与 主串 直接 进行 匹配 （窗口中的子串的向量已经计算好了）

        time = 0

        for pattern in pattern_list:

            print(pattern)  # TODO: 提交 记得 注释

            pattern_mid_vector = self._string_to_vector(pattern[1:pattern_length - 1])  # 预先计算：将 pattern 中间（去除首尾字符）的 字符串 向量化


            i = 0
            while i <= (len(main_string) - pattern_length):  # 滑动窗口

                sub_main_string = main_string[i:i + pattern_length]  # 主串在 窗口中的子串

                if sub_main_string[0] == pattern[0] and sub_main_string[-1] == pattern[-1]:  # 首尾 字母是否相同

                    if pattern_mid_vector == main_string_vector_cache[i]:  # TODO： 比较 两个  tuple 是否相等 a=(2,1,0,...,0) b=(2,1,0,...,1) a==b? 在 python 中是一位一位比较
                                                                          # https://stackoverflow.com/questions/5292303/how-does-tuple-comparison-work-in-python
                        time += 1
                        break  # 匹配 一个 即可退出

                i += 1

        return time

if __name__ == '__main__':

    sol=solutions()

    # print(sol._hash_string('aabc'))
    # print(sol._hash_string('abca'))

    # print(sol._match_one_pattern('zbddb','zbbdbbzxtndrvjblnzjfpvhdhhpxjdnabd'))

    # print(sol._string_to_vector('abbc'))


    # IDE 测试 阶段：
    test=Test()
    # test.test_small_dataset(sol.scrambled_words)
    # test.test_large_dataset(sol.scrambled_words)

    # from vprof import runner
    #
    # runner.run(test.test_large_dataset, 'cmhp', args=(sol.scrambled_words), host='localhost', port=8000)

    # 提交 阶段：

    # pycharm 中打开命令行 窗口 ，测试数据在 inputs ，< 为输入重定向
    # 1. cd 到 目标文件夹
    # 2. 执行：
    #   E:\python package\python-project\Beauty_of_data_structure_and_algrithms\kickstart\2018 Round A> python Scrambled_Words.py < inputs
    #   CMD 输出：Case #1: 4
    #
    #   python Scrambled_Words.py < ref/C-small-practice.in
    #   python Scrambled_Words.py < ref/C-large-practice.in

    T = int(input()) # 一共T个 测试数据

    for t in range(1, T + 1):

        L=int(input()) # 字典中词的个数

        dictionary = [ s for s in input().split(' ')]  # 字典 的 单词列表

        s1,s2,N,A,B,C,D = [ i for i in input().split(' ')]

        main_string=test.building_main_string(s1,s2,int(N),int(A),int(B),int(C),int(D))

        res = sol.scrambled_words(dictionary, main_string)

        print('Case #{}: {}'.format(t, res))



    # 性能测试
    # vprof -c h Scrambled_Words.py






