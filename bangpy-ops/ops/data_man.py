import bangpy

class data_man:
    bp = None

    def init(self, bp):
        self.bp = bp
        self._current_core_start = self.bp.Scalar(bangpy.int32, "current_core_start")
        self._current_core_end = self.bp.Scalar(bangpy.int32, "current_core_end")
        self._total_count_in_core = self.bp.Scalar(bangpy.int32, "total_count_in_core")

    def calc_core_process_count(self, data_total_len, task_num):
        one_core_count = self.bp.Scalar(bangpy.int32, "one_core_count")
        one_core_count.assign(data_total_len // task_num)

        remain = self.bp.Scalar(bangpy.int32, "remain") 
        remain.assign(data_total_len % task_num)

        with self.bp.if_scope(self.bp.taskId < remain): #如果存在余数 将其均摊给各核   taskId从0起
            self._current_core_start.assign((one_core_count + 1) * self.bp.taskId )
            self._current_core_end.assign((one_core_count + 1) * (self.bp.taskId + 1) - 1) #此处应该不需要减1 待验证  python切片会自动将上标减1
        with self.bp.else_scope():
            self._current_core_start.assign((one_core_count + 1) * remain + one_core_count * (self.bp.taskId - remain))
            self._current_core_end.assign((one_core_count + 1) * remain + one_core_count * (self.bp.taskId - remain) + one_core_count - 1) 

        self._total_count_in_core.assign(self._current_core_end - self._current_core_start + 1)