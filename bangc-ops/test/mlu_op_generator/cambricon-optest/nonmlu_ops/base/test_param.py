class TestParam:
    def __init__(self, *args, **kwargs):
        self.loop_times_ = 1000
        self.warm_up_times_ = 100


class TestData:
    def __init__(self, *args, **kwargs):
        self.threshold_list_ = []
        self.latency_ = 0
        self.workspace_size_ = 0
