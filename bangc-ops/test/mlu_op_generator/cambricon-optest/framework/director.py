from .builder import MluOpBuilder


class Director:
    def __init__(self, test_type, *args, **kwargs):
        self.test_type_ = "MluOp"
        self.args_ = args
        self.kwargs = kwargs

    def run(self):
        if self.test_type_ == "MluOp":
            MluOpBuilder(*self.args_, **self.kwargs).run()
        else:
            raise Exception("Director test type not support")
