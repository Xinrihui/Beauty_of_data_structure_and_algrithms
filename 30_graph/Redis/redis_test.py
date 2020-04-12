from redis import *


POOL=ConnectionPool(host='localhost',port=6379,max_connections=100) # 创建 连接池

r = Redis(connection_pool=POOL) # 从连接池 拿到连接

# r.set('name', 'runoob')  # 设置 name 对应的值
# a=r.get('name') # 取出键 name 对应的值


# r.zadd('grade', {'bob':100, 'mike':99, 'lucy':87})

# print( r.zrank("grade", 'lucy')) # 0  返回key为 'lucy'  zset中 元素的排名（按score从小到大排序）即下标
# print( r.zrevrank("grade", 'lucy')) # 2  返回key为 'lucy' 的zset中元素的倒数排名（按score从大到小排序）即下标
#
# print( r.zrange("grade", 0,0,withscores=True)) # 按照 标号 的范围 找 有序集合 中的 元素 [(b'lucy', 87.0)]
# print( r.zrange("grade", 1,1,withscores=True)) # [(b'mike', 99.0)]
# print( r.zrange("grade", 2,2,withscores=True)) # [(b'bob', 100.0)]
#
# print(r.zrangebyscore("grade", 99, 99, withscores=True))   # [(b'mike', 99.0)]  按照分数 的范围 获取有序集合中元素
#
# print(r.zrangebyscore("grade", 101, 101, withscores=True)) # []
#
# print(r.zcard("grade")) # 返回 zset 的元素个数
#
#
# for i in r.zscan_iter("grade"): # 遍历迭代器
#     print(i)
#
# print(r.zrange("grade1", 2,2,withscores=True)) #[]


# r.zadd('follow', {'a':1, 'b':2, 'c':3}) # score 必须为 数字
# print(r.zrangebylex('follow',min='-',max='+',start=0,num=2)) #按照字典序 输出 Key

