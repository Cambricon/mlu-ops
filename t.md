**问题描述：**
数据从SRAM拷到GDRAM时，打印查看后发现并没有拷进去。

**期望的代码行为：**
(''')\
with self.tcp.block("data_copy"):\
    self.tcp.memcpy(buffer_out_s[begin:end], buffer_io_n)\
    with self.tcp.if_scope(j==2):\
        st.assign(start-2*data_each_time)\
        # with self.tcp.if_scope(tcp.all(task_id==0,i==2)):\
        #     self.tcp.print(buffer_out_s[begin:begin+data_each_time])\
              self.tcp.memcpy(buffer_out[st:stop],buffer_out_s[begin-2*data_each_time:end])\
              self.tcp.sync_cluster()\
        # with self.tcp.if_scope(tcp.all(task_id==0,i==2)):\
        #     self.tcp.print(buffer_out[st:st+data_each_time])\
(''')\
期望buffer_out的值与buffer_out_w里面的值一致。

**实际运行的代码行为：**
buffer_out初始值为全0：\
‘data_out_dev = bangpy.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)’\
拷贝后输出的仍然是全0，也即数据没有拷入进去

**运行环境：**
代码分支：[分支链接](https://github.com/pingmu123/mlu-ops.git)\
bug复现步骤：定位到hard_sigmoid.py文件184-193行，修改相关print语句，然后直接python3 mytest.py即可\
bangpy版本：1.3.1\
cncc：v3.6.1
