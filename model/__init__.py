# coding:utf-8
'''
    author : Cindy
    Date : 2017/11/7
    代码目的：
    
'''


import time


def timeDecor(func):
    # 一个用于统计函数运行时间的装饰器

    def innerDef(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        t = t2 - t1
        print ("{0} 函数部分运行时间 ：{1}s".format(str(func.__name__), t))
        return result

    return innerDef


