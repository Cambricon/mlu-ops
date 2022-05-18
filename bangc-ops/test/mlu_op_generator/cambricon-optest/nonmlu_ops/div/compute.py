from nonmlu_ops.base import *
import numpy as np
import tensorflow as tf


@registerTensorList("div")
class DivTensorList(TensorList):
    pass


@registerOp("div")
class DivOp(OpTest):
    def __init__(self, tensor_list, params):
        super().__init__(tensor_list, params)
        self.output_shape_ = self.tensor_list_.getOutputTensor(0).getShape()

    def compute(self):
        tf.reset_default_graph()

        input_tensor0 = self.tensor_list_.getInputTensor(0)
        input_tensor1 = self.tensor_list_.getInputTensor(1)
        out_tensor = self.tensor_list_.getOutputTensor(0)

        input0 = DataNode("float32")
        input1 = DataNode("float32")
        result_fp32_node = DataNode("float32")

        quantize_utils.caseDataNode(input0, input_tensor0.getDataNode())
        quantize_utils.caseDataNode(input1, input_tensor1.getDataNode())

        output_node = DataNode(out_tensor.getDataType().getStr())

        tf_input0 = tf.placeholder(np.float32)
        tf_input1 = tf.placeholder(np.float32)

        div_op = tf.divide(tf_input0, tf_input1, name="divOp")
        with tf.Session() as sess:
            result = sess.run(div_op, feed_dict={
                              tf_input0: input0.getData(), tf_input1: input1.getData()})
            result_fp32_node.setData(result)

        quantize_utils.caseDataNode(output_node, result_fp32_node)
        out_tensor.setData(output_node.getData())


@registerProtoWriter("div")
class OpTensorProtoWriter(MluOpProtoWriter):
    pass
