精度特性
================

算子精度变化说明
----------------

在相同的硬件架构下，Cambricon CNNL对于部分算子能保证软件版本的更新不会影响算子的精度（bit-wise reproducibility）。此类算子包括：

- cnnlOpTensor
- cnnlAddTensor

在相同的硬件架构下，Cambricon CNNL对于少数算子不能保证软件版本的更新不会影响算子的精度。此类算子包括：

- Convoltion类算子，如cnnlConvolutionForward、cnnlConvolutionBackwardFilter。
- Pooling类算子，如cnnlPoolingForward。
- Reduce类算子，如cnnlBatchnormForward。
- 激活类算子，如cnnlActivationForward。

此外，不同算子使用不同的计算方法也可能会导致算子精度的变化。这是由于某些算法内部实现时，采用了更复杂的计算方式或者更多的空间来保证算子精度。

.. _算子数值修约模式:

算子数值修约模式
----------------

Cambricon CNNL算子在设备端实现浮点到整型的数据转换过程中，提供了两种数值修约（Rounding）模式，并分别提供彼此互斥的软件包。

对于训练场景的应用，两个版本Cambricon CNNL都可以使用，推荐使用默认版本（Round half away from zero）。对于推理场景的应用，用户应根据推理模型采用的修约模式适配对应的Cambricon CNNL版本。


Round half away from zero
>>>>>>>>>>>>>>>>>>>>>>>>>

远离零点四舍五入，默认版本Cambricon CNNL算子采用的修约模式，与C语言 ``round()`` 函数采用的修约模式相同。该模式对于正半轴数向上四舍五入，对于负半轴数向下四舍五入。

.. math::

   \begin{aligned}
   Round(x)=sgn(x) \lfloor \lvert x \rvert + 0.5 \rfloor=
    \begin{cases} \lfloor x + 0.5 \rfloor, & x \ge 0 \\ - \lfloor -x + 0.5 \rfloor, & x < 0 \end{cases}
   \end{aligned}

如-114.5可修约为-115，-9.5可修约为-10，114.5可修约为115。

Round half up
>>>>>>>>>>>>>>>>>>>>>>>>>

向上（正无穷）四舍五入，软件包名含 ``-round-half-up`` 采用的修约模式。该模式无论正负数一律向正无穷方向四舍五入。

.. math::

   \begin{aligned}
   RoundHalfUp(x)= \lfloor x+0.5 \rfloor= \lceil { \frac{\lfloor 2x \rfloor}{2} } \rceil
   \end{aligned}

如-114.5可修约为-114，-9.5可修约为-9，114.5可修约为115。
