##---- python tips----##

#1.循环遍历删除满足条件的元素

def delete_element(arr):
    for ele in range(len(arr)-1,-1,-1):
        if arr[ele] ==2:
            del arr[ele]
    return arr

# print (delete_element([1,2,3,2,5]))


#2. 数组切片的赋值问题
def test_1():
    a=[1,2,3]
    a=a[0:2]+[5]+a[1:]
    print(a)

    b=[1,3,5]
    # b.insert(index=1,object:2)
    b.insert(1,2)
    print(b)

    c=[1,3,5]
    c.extend([6])
    print(c)

# test_1()

#2.bisect 的使用
import  bisect

def test2():
    l = [10, 30, 50, 70]
    print(bisect.bisect_left(l,30))
    print(bisect.bisect_left(l, 20))
    print(bisect.bisect_left(l, 80))

    # l.pop()
    # print(l)

    # print('index: ', l.index(70,0,3))
    print('l length: ',len(l))
    print('index: ', l.index(70, 0, 3+1))
    # l.remove(10)
    # print(l)

# test2()
from collections import deque

from queue import Queue

def test3():

    l=[99,203]
    l.insert(1,105)
    print(l)



    l=deque([99,203])
    print(l)
    # print(l[0:])

    # q = Queue([99,203])
    # print(q.get())

test3()

#3. 二分查找的三种实现:
# http://kuanghy.github.io/2016/06/14/python-bisect
def binary_search_recursion(lst, value, low, high):
    if high < low:
        return None
    mid = (low + high) // 2
    if lst[mid] > value:
        return binary_search_recursion(lst, value, low, mid-1)
    elif lst[mid] < value:
        return binary_search_recursion(lst, value, mid+1, high)
    else:
        return mid

def binary_search_loop(lst,value):
    low, high = 0, len(lst)-1
    while low <= high:
        mid = (low + high) // 2
        if lst[mid] < value:
            low = mid + 1
        elif lst[mid] > value:
            high = mid - 1
        else:
            return mid
    return None

def binary_search_bisect(lst, x):
    from bisect import bisect_left
    i = bisect_left(lst, x)
    if i != len(lst) and lst[i] == x:
        return i
    return None


# import random
# random.seed(1)
# lst = [random.randint(0, 10000) for _ in range(100000)]
# lst.sort()
#
# def test_recursion():
#     binary_search_recursion(lst, 999, 0, len(lst)-1)
#
# def test_loop():
#     binary_search_loop(lst, 999)
#
# def test_bisect():
#     binary_search_bisect(lst, 999)
#
# import timeit
# t1 = timeit.Timer("test_recursion()", setup="from __main__ import test_recursion")
# t2 = timeit.Timer("test_loop()", setup="from __main__ import test_loop")
# t3 = timeit.Timer("test_bisect()", setup="from __main__ import test_bisect")
#
# print("Recursion:", t1.timeit())
# print("Loop:", t2.timeit())
# print("bisect:", t3.timeit())



##---- end python tips----##
