# -*- coding: UTF-8 -*-
from collections import *

import heapq


class Priority_Queue(object):
    """
    
    by XRH 
    date: 2020-05-02 

    """

    def __init__(self, initial=None, key_func=lambda x: x, compare_key_func=lambda x: x):
        """
        利用堆实现 优先队列
        提供以下功能：
        1.指定主键，可以根据主键直接访问 
        2.指定用来 比较大小的 键 
        
        :param initial: 传入堆的 初始 list，eg.[('a',1),('b',2),...]
        :param key_func: 主键 key 
        :param compare_key_func: 比较大小的键 value
        """

        self.key_func = key_func

        self.compare_key_func = compare_key_func

        self.length = len(initial)

        self.hash_table = {}
        self._data = []

        if initial:

            for item in initial:  # [(key1,value),(key2,value)]

                p_index = [compare_key_func(item), item]  # p_index 是一个指针，指向了 list 所在的内存地址

                self.hash_table[key_func(item)] = p_index
                self._data.append(p_index)

            heapq.heapify(self._data)

        else:
            self._data = []

    def __len__(self):

        return self.length

    def has_Key(self, key):
        """
        判断 Key 是否存在
        :param key: 
        :return: 
        """
        return key in self.hash_table

    def get_byKey(self, key):
        """
        通过 key 读取对应的元组
        :param key: 
        :return: 
        """
        if key in self.hash_table:
            return self.hash_table[key]
        else:
            return None

    def push(self, item):
        """
        插入一个 元组，然后调整堆
        :param item: 
        :return: 
        """
        self.length += 1

        p_index = [self.compare_key_func(item), item]

        self.hash_table[self.key_func(item)] = p_index
        heapq.heappush(self._data, p_index)

    def pop(self):
        self.length -= 1

        ele = heapq.heappop(self._data)
        self.hash_table.pop(self.key_func(ele[1]))

        return ele[1]

    def update_byKey(self, key, value):
        """
        通过 key 找到对应的元组，并更新它的值，同时调整堆

        :param key: 
        :param value: 
        :return: 
        """
        # self.hash_table[key]=[key,(key,value)] # 无法更新 堆中 被引用的 List

        self.hash_table[key][0] = value  # self.hash_table[key] 返回为 List（它也 在堆中被引用） 的内存地址 ，self.hash_table[key][0] 直接更改了 List 中的内容
        self.hash_table[key][1] = (key, value)

        heapq.heapify(self._data)


if __name__ == '__main__':


    heap = Priority_Queue([(0, 0)], key_func=lambda x: x[0], compare_key_func=lambda x: x[1])  #

    heap.push((1, 2))
    heap.push((2, 3))

    print(heap.get_byKey(1))

    print(heap.pop())

    heap.update_byKey(1, 4)
    print(heap.pop())






