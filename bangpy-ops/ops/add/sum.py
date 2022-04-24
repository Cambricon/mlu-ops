import numpy as np
import time

# a = np.ones(shape=268435456,dtype=np.float32)
# start = time.time()
# a.sum()
# end = time.time()
# print("total:",end-start)

# #保证能被2除
# #保证128对齐 就是32 
# current = end_index //2 
# current%32
length = 512
a = []
for i in range(length):
    a.append(i)

行数要是128字节对齐元素个数得倍数




不够或多余得 用一维得进行运算