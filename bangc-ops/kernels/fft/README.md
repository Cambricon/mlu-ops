# Introducation

在这个目录中，我们将整合所有与FFT算子相关的代码，代码支持了FFT算子四种模式：rfft(r2c), irfft(c2r), fft(c2c), ifft(c2c)；同时，出于性能考虑，在具体实现中，不同规模会调用不同的算法，共有三种：DFT、FFT_cooley-tukey和FFT_stockham，具体的代码组织方式如下：

## 代码目录以及说明

1.目录的树状图如下：
  ├── c2c_fft
  │   ├── c2c_fft.h
  │   └── c2c_fft_host.mlu
  ├── common
  │   ├── fft_basic_ops.cpp
  │   ├── fft_basic_ops.h
  │   ├── fft_common_kernels.h
  │   └── fft_common_kernels.mlu
  ├── fft.h
  ├── fft.mlu
  ├── fft_optm_device
  │   ├── fft_cooley-tukey_ux_device.mlu
  │   └── fft_stockham_u1_device.mlu
  ├── irfft
  │   ├── irfft.h
  │   └── irfft_host.mlu
  └── rfft
      ├── rfft.h
          └── rfft_host.mlu

2.fft.h和fft.mlu：
   * fft.h：文件中定义了一些基本的结构体，如：不同模式、策略、地址等；进行了golbal函数的声明；
   * fft.mlu：文件中定义了用户调用的公共接口，如：策略初始化、workspace初始化、host函数选择、基本防呆操作等；每一种模式都会先进入到这个文件，然后根据判断结果，调用对应模式的host代码；

3.common文件夹：
   * fft_basic_ops.h：在进行FFT调用时，也会使用到别的接口，如转置、量化、矩阵乘等，这些接口的函数调用封装的声明均放置在这个文件；还有一些封装的基本公共函数也放在这里：如findLimit函数；
   * fft_basic_ops.cpp：给出fft_basic_ops.h中声明接口的实现；
   * fft_common_kernels.h：生成W矩阵通常是一个耗时的操作，在网络训练中，只有第一次迭代时会生成一次，这里进行了预生成W矩阵接口函数的声明；
   * fft_common_kernels.mlu：给出fft_common_kernels.h中声明接口的实现；

4.rfft文件夹：
   * rfft.h：给出rfft的策略函数、workspace空间申请函数和执行函数等的声明；
   * rfft_host.mlu：rfft host函数的具体实现；会根据策略函数的结果选择：DFT、FFT_cooley-tukey或FFT_stockham算法；

5.irfft文件夹：
   * 文件夹结构同rfft，只是针对irfft的声明和实现；

6.c2c_fft文件夹：
   * 文件夹结构同rfft，只是针对fft和ifft的声明和实现，因为两者差别只有一个常数因子，所以放在了同一个文件夹中。

7.fft_optm_device文件夹：
   * fft_cooley-tukey_ux_device.mlu：优化kernel device代码，基于cooley-tukey算法思想实现；
   * fft_stockham_u1_device.mlu：优化kernel device代码，基于stockham算法思想实现；
   * 备注：DFT调用mluOpTranspose, mluOpMatmul等kernel实现，调用fft_basic_ops.cpp中封装好的函数即可，未单独实现kernel device代码。

