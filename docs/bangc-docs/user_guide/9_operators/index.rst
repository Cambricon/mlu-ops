.. _算子列表:

算子支持
==========================

下面具体介绍Cambricon BANGC OPS支持的算子及其功能描述。有关算子详情，请参见《Cambricon BANGC OPS Developer Guide》。

mluOpAbs
--------

返回绝对值。

公式如下：

.. math::

     y_i = |x_i|

其中：

- ``i`` 表示一个多元组的索引，例如在4维时可以表示（n,c,h,w）。
- ``xi`` 和 ``yi`` 表示多元组中 ``i`` 索引处的元素。

mluOpLog
-----------------------------

计算输入张量的以e、2、10为底的对数。

log的计算公式为：

.. math::

     y_i = log(x_i)

log2的计算公式为：

.. math::

   y_i = log2(x_i)


Llg10的计算公式为：

.. math::

   y_i = log10(x_i)


注：

- ``i`` 表示一个多元数组的索引，表示多维张量。
- :math:`x_i`、:math:`y_i` 表示多元组中 i 索引处的元素。

mluOpDiv
-----------------------------

两个输入张量相除，得到输出结果。

公式如下：

.. math::

   z_i = x_i/y_i

其中：

- ``i`` 表示一个多维数组的索引，表示多维张量，例如在4维时可以表示(n,c,h,w)。
- ``xi``、``yi``、``zi`` 表示多维数组中 ``i`` 索引处的元素。

mluOpPolyNms
----------------------------
计算不规则四边形的非极大值抑制，用于删除高度冗余的不规则四边形输入框。


mluOpGenerateProposalsV2
----------------------------
generate_proposals_v2根据每个检测框为 foreground 对象的概率 scores ，使用非极大值抑制来推选生成用于后续检测网络的ROIs，其中的检测框根据anchors和bbox_deltas计算得到。该算子是generate_proposals 的第二个版本。

mluOpPriorBox
---------------------------
prior_box为SSD（Single Shot MultiBox Detector）算法生成候选框。具体是在输入input的每个位置产生num_priors个候选框。候选框的坐标为（x1,y1,x2,y2），代表候选框的左上和右下的点的坐标。总共生成 boxes_num = height * width * num_priors 个候选框，其中：

一个点生成的num_priors个候选框的中心都一样，默认为每个网格的中心，offset为候选框的中心位移。

例如，（0,0）处的候选框中心点为（0+offset，0+offset）。

每个点生成的第j（0<j<=num_priors）个候选框之间对应的宽，高都一样（对超出边界的候选框不裁剪的前提下）。

例如，第一个点生成的第1个候选框和第二个点生成的第1个候选框的宽高相等。

mluOpPsRoiPoolForward
---------------------------
一种针对位置敏感区域的池化方式。psroipool的操作与roipool类似，不同之处在于不同空间维度输出的图片特征来自不同的feature map channels，且对每个小区域进行的是Average Pooling，不同于roipool的Max Pooling。对于一个输出 k * k 的结果，不同空间维度的特征取自输入feature map中不同的组，即将输入的feature map在通道维度均匀分为k * k组，每组的channel数与输出的channel一致。

mluOpPsRoiPoolBackward
---------------------------
mluOpPsRoiPoolForward算子的反向。

mluOpRoiCropForward
---------------------------
根据感兴趣区域提取固定大小的输出特征。从输入的 grid 中提取一个 (y, x) 坐标映射参数，反映射到 input 中的 A 处得到坐标信息(Ax, Ay)，获取A点附近整数点位 top_left, top_right, bottom_left, bottom_right 四处像素值，根据 grid 中每个像素位 bin 的索引获得 output 中对应的偏移地址，最后通过双线性插值计算输出 output 的像素值。

mluOpRoiCropBackward
---------------------------
mluOpRoiCropForward算子的反向。

mluOpSqrt
-----------

开方的操作。

公式如下：

.. math::

   y_i = \sqrt{x_i}

其中：

- ``i`` 表示一个多维数组的索引，表示多维张量，例如在4维时可以表示 (n,c,h,w)。
- :math:`x_i` 和 :math:`y_i` 表示多元组中 i索引处的元素。

mluOpSqrtBackward
-------------------

计算 Sqrt 的导数。

假设输入为 x，输出为 y，上一层回传的导数为 :math:`diff_y`，公式如下：

.. math::

   diff_x = 0.5 * \frac{diff_y}{y}


mluOpYoloBox
-------------------
yolo_box 负责从检测网络的 backbone 输出部分，计算真实检测框 bbox 信息。该算子三个输入 tensor，两个输出 tensor，输入 x 维度 [N, C, H, W]，输入 img_size 维度 [N, 2]，输入 anchors 维度 [2*S]，其中S表示每个像素点应预测的框的数量，输出 boxes 维度 [N, S, 4, H*W]，输出 scores 维度 [N, S, class_num, H*W]。

mluOpThreeInterpolateForward
-------------------
three_interpolate_forward 对三个输入特征做加权线性插值获得目标特征。其中三个输入特征在 features tensor 中的下标由 indices tensor 决定，将选择出来的三个输入特征乘上对应的 weights tensor 中的权重，并将对应的乘法结果进行累加得到目标特征，对于每个 batch，在每个 channel 上重复上述过程 N 次就得到加权插值后的输出结果。该算子有三个输入 tensor，一个输出 tensor,输入 features 维度 [B, C, M]，输入 indices 维度 [B, N, 3]，输入 weights 维度 [B, N, 3]，输出 output 维度 [B, C, N]。

mluOpBallQuery
-------------------
ballquery 负责返回球域内点的 indexes。该算子有五个输入和一个输出，其中输入有 new_xyz 和 xyz 两个 tensor 以及 min_radius、max_radius、nsample 三个标量，有 idx 一个输出 tensor。其以 new_xyz 中的点为球心，以 min_radius 和 max_radius 分别为内径和外径， 返回球域内前 nsample 个 xyz 点的index。其中，输入 new_xyz 维度 [B, M, 3]，输入 xyz 维度 [B, N, 3], 输入 min_radius、max_radius、nsample 为标量，输出 idx 维度 [B, M, nsample]。

mluOpCopy
-------------------
该算子主要在语音网络中使用，对数据块进行 device 到 device 的拷贝。

mluOpExpand
-------------------
该算子应用于各种需要广播的场景，实现张量的维度扩展。算子需要输出维度与输入维度符合广播扩展标准，根据输入输出的维度，将输入数据复制并扩展成输出维度。
