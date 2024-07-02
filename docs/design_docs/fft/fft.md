# FFT1d 算子开发设计方案


* #### 文档基本信息
| 算子名称                                   |
| ------------------------------------------ | 
| FFT(RFFT1d, IRFFT1d, FFT1d, IFFT1d)   | 

* #### 内容描述

本文档为`FFT`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

* #### 算子需求checklist

算子需求提出者需要`提供`的信息如下：

- 算子接口描述：实现RFFT1d, IRFFT1d, FFT1d, IFFT1d的DFT算法
- 功能描述：实现RFFT1d, IRFFT1d, FFT1d, IFFT1d的傅里叶变换算法
- 是否需要支持原位：需要*
- 是否需要支持stride机制：需要

*当支持原位的时候，需要保证input的sequence长度补齐到output的sequence的长度，否则会出现未定义的结果。

算子需求提出者需要`check`的部分如下：

- 1.1 算子需求分析
- 1.2 算子功能和应用场景描述
- 1.3 算子输入输出参数要求
- 1.4 算子限制
- 1.5 验收标准
- 2.2 接口设计
- 3.5 测试用例（需求提出者check算子需求表中所给规模是否列出）

## 1 需求分析

### 1.1 算子需求分析

| 算子功能简介| 对数列进行傅里叶变换操作，详细描述在1.2中进行说明            |
|-------------|--------------------------------------------------------------|
| 需求来源    | PyTorch/Tensorflow                                       |
| 应用网络    | Conformer                                        |
| 输入数据类型| half, float, complex_half, complex_float          |
| 输入Shape   | [batches, array] |
| 输入Layout  | ARRAY                             |
| 输出数据类型| half, float, complex32, complex64                         |
| 输出Shape   | [batches, array]                   |
| 输出Layout  | ARRAY                                                    |
| 模式(可选） |                                                              |
| 是否含有dim/axis等类似语义的参数且该参数支持负数/其他特殊处理 | 通过stride语义来支持dim |
| 是否含有labels/index等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 无 |
| 是否需要支持原位        | 是                                                  |
| 是否需要支持stride机制  | 是                                                  |
| 是否需要支持广播  | 否                       |
| 0元素检查是否直接返回  | 否(array=0时不支持，支持以下两种情况：1.batch等于0；2.输入或者输出的dim等于0，但是补齐到array)                               |
| 其他特殊需求(在线量化，融合，转数提前等，可选)| 无 |
| 本次开发优先支持的规模/模式| 支持rfft,irfft,fft,ifft |

### 1.2 算子功能和应用场景描述

RFFT: 对一个长度为N的实数数列进行傅里叶变换，输出长度为 N/2+1的复数数列。因为后半部分结果和前半部分是复共轭的关系，所以该算子仅输出前半部分结果。

IRFFT: RFFT的反向，对一个长度为N/2+1的复数数列进行傅里叶变换，输出长度为 N的实数数列。因为后半部分输入和前半部分是复共轭的关系，所以该算子仅提供前半部分输入，需自行补齐。

FFT: 对一个长度为N的复数数列进行傅里叶变换，输出长度为 N的复数数列。

IFFT: FFT的反向，对一个长度为N的复数数列进行傅里叶变换，输出长度为 N的复数数列。

公式：
```math
X[k]=\Sigma_n W_N^{kn}x(k)   \\

W_N^{kn}=e^{sign*ikn\frac{2\pi}{N}},
sign= \left \{
\begin{array}{ll}
-1, fft\  or \ rfft \\
1, ifft\  or \ irfft
\end{array}
\right.
```
备注：

1、需要说明对nan/inf的特殊处理，输入存在nan/inf时，输出为按照IEEE754标准根据FFT计算公式得到，其中任何数乘nan为nan，0乘inf为nan，非0乘正负inf根据符号为正负inf，任何数加nan为nan，任何非nan数加正负inf根据符号为正负inf。
在部分情况下，pytorch的CPU结果和GPU结果并不是严格按照IEEE754标准和FFT计算公式得到，因此无法与MLU计算结果对齐。例如，某个batch的输入数据包含一个inf，且inf所在位置对应的系数不为0，此时GPU计算结果全为nan，但是MLU计算结果全为inf。
再例如，某个batch的输入部分包含一个nan，且nan所在位置对应的系数为0，此时GPU计算结果为0，但是MLU计算结果仍为nan。

2、N为大质数的情况，可能来不及支持。



### 1.3 算子输入输出参数要求

| 参数        | 语义 | 类型（输入/输出） | 支持类型    | 物理布局 | 规模限制         |
| ----------- | ---- | ----------------- | ----------- | -------- | ---------------- |
| handle      |      | 输入              |             | /        | 无               |
| fft1d_plan |      | 输入              |             | /        | 暂时不支持大质数 |
| input      |      | 输入              | half, float， complex_half, complex_float | ARRAY    | 无               |
| output      |      | 输出             | half, float， complex_half, complex_float | ARRAY    | 无               |

## 2 算子接口设计

### 2.1 参考接口

- TensorFlow

```python
#Computes the 1-dimensional discrete Fourier Transform of a real-valued signal over the inner-most dimension of input.
# Since the DFT of a real signal is Hermitian-symmetric, RFFT only returns the `fft_length / 2 + 1` unique components of the FFT: The zero-frequency term, followed by the `fft_length / 2` positive-frequency terms.
# Along the axis RFFT is computed on, if `fft_length` is smaller than the corresponding dimension of input, the dimension is cropped. If it is larger, the dimension is padded with zeros
tensorflow.signal.rfft(input_tensor, fft_length=None, name=None)

tensorflow.signal.irfft(input_tensor, fft_length=None, name=None)

tensorflow.signal.fft(input_tensor, name=None)

tensorflow.signal.ifft(input_tensor, name=None)
```

- PyTorch
```python
#Computes the one dimensional Fourier transform of real-valued input.
# The FFT of a real signal is Hermitian-symmetric, X[i] = conj(X[-i]) so the ouput contains only the positive frequencies below the Nyquist frequency. To compute the full output, use fft()
# input(Tensor) - the real input tensor
# s(Tuple[int], optional) - Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the real FFT
# dim(Tuple[int], optional) - The dimension along which to take the one dimensional real FFT.
# norm(str, optional) - Normalization mode. For the forward transform(rfft()), these correspond to:
# 	"forward" - normalized by `1\n`
# 	"backward" - no normalization
#   "ortho" - normalize by `1/sqrt(n)` (making the FFT orthonormal)
# Calling the backward transform (irfft()) with the same normalization mode will apply an overal normalization of 1/n between the two transforms. This is required to make irfft() the exact inverse
# Default is "backward" (no normalization).
torch.fft.rfftn(input, s=None, dim=None, norm=None, *, out=None)->Tensor

torch.fft.irfftn(input, s=None, dim=None, norm=None, *, out=None)->Tensor

torch.fft.fftn(input, s=None, dim=None, norm=None, *, out=None)->Tensor

torch.fft.ifftn(input, s=None, dim=None, norm=None, *, out=None)->Tensor
```

- Mathematica

```mathematica
Fourier[list]
(*finds the discrete Fourier transform of a list of complex numbers*)
Fourier[list, {p1, p2, ...}]
(*returns the specified positions of the discrete Fourier transform*)
```

- Matlab

```matlab
# computes the discrete Fourier transform of X using a fast Fourier transform algorithm
# if X is a vector, then fft(X) returns the Fourier transform of the vector
Y = fft(X)
# returns the n-point DFT. If no value is specified, Y is the same size as X.
# If X is a vector, and the length of X is greater than n, then X is truncated to length n.
Y = fft(X, n)
# returns the Fourier transform along the dimension dim. For example, if X is a matrix, then fft(X, n, 2) returns the n-point Fourier transform of each row
Y = fft(X, n, dim)
```

- OTFFT

```c
OTFFT::RFFT::fwd(const_double_vector x, complex_vector y);
```

- FFTW

```c
fftw_plan fftw_plan_dft_r2c_1d(int n0, double *in, fftw_complex *out, unsigned flags);

fftw_plan fftw_plan_many_dft_r2c(int rank, const int *n, int howmany, double *in, const int *inembed, int istride, int idist, fftw_complex *out, const int *onembed, int ostride, int odist, unsigned flags);

void fftw_execute_dft_r2c(const fftw_plan p, double *in, fftw_complex *out);

void fftw_execute_dft_c2r(const fftw_plan p, fftw_complex *in, double *out);

void fftw_execute_dft(const fftw_plan p, fftw_complex *in, fftw_complex *out);

void fftw_destroy_plan(fftw_plan plan);
```



### 2.2 接口设计

```c++
/*! The descriptor of FFT (Fast Fourier Transform) operator that holds FFT information including
 * the tensor descriptor of input tensor and output tensor, the rank of FFT, the FFT size on each
 * dimension, the size of reserved space and the size of workspace.
 *
 * You need to call the ::mluOpCreateFFTPlan function to create a descriptor for the FFT operator, and call
 * the ::mluOpMakeFFTPlanMany function to set the information of the FFT operator to the descriptor.
 * Then, you need to allocate the reserved space and set the space to the fft descriptor by ::mluOpSetReserveArea.
 * Also, you need to destroy the MluOp context at the end with the ::mluOpDestroyFFTPlan.
 */
typedef struct mluOpFFTStruct *mluOpFFTPlan_t;

/*!
 *  @brief Creates a descriptor pointed by \b fft_plan for the FFT operator, and allocates memory
 *  for holding the information about the FFT operation. The information is defined in ::mluOpFFTPlan_t.
 */
mluOpStatus_t mluOpCreateFFTPlan(mluOpFFTPlan_t *fft_plan);

/*!
 *  @brief Initializes the FFT descriptor pointed by \b fft_plan that is previously created
 *  with the ::mluOpCreateFFTPlan function, and sets the information about the
 *  tensor descriptors of input tensor and output tensor, the rank of FFT, and the FFT size on each
 *  dimension.
 *
 *  This function also gets the size of MLU memory buffers for FFT execution, including \b reservespace_size and
 *  \b workspace_size. The size of extra workspace is based on the given information of the
 *  \b fft_plan.
 */
mluOpStatus_t mluOpMakeFFTPlanMany(mluOpHandle_t handle,
                                   mluOpFFTPlan_t fft_plan,
                                   const mluOpTensorDescriptor_t input_desc,
                                   const mluOpTensorDescriptor_t output_desc,
                                   const int rank,
                                   const int n[],
                                   size_t *reservespace_size,
                                   size_t *workspace_size);
/*!
 *  @brief Bond the reserve space to the \b fft_plan. The size of reserved space can be derived through ::mluOpMakeFFTPlanMany.
 */
mluOpStatus_t mluOpSetFFTReserveArea(mluOpHandle_t handle,
                                   mluOpFFTPlan_t fft_plan,
                                   void *reservespace);
/*!
 *  @brief Executes any FFT. In case of complex-to-real and real-to-complex
 *  transforms \b direction parameter is ignored. This function stores the Fourier coefficients
 *  in the output array. If the address of input and output are the same, an in-place FFT
 *  is adopted.
 */
mluOpStatus_t mluOpExecFFT(mluOpHandle_t handle,
                           const mluOpFFTPlan_t fft_plan,
                           const void *input,
                           const float scale_factor,
                           void *workspace,
                           void *output,
                           int direction);

/*!
 *  @brief Destroys a FFT plan \b fft_plan that is created with the
 *  ::mluOpCreateFFTPlan function.
 */
mluOpStatus_t mluOpDestroyFFTPlan(mluOpFFTPlan_t fft_plan);
```

框架使用场景，下面假设一个一维rfft，batch为2000，n=400的rfft：

1. 建立fft描述符

   ```c
   mluOpFFTPlan_t fft_plan;
   mluOpCreateFFTPlan(&fft_plan);
   ```

2. 给fft描述符设定参数，并获取reserve_size，workspace_size大小

   ```c
   mluOpTensorDescriptor_t input_desc, output_desc;
   mluOpDataType_t input_data_type = MLUOP_DTYPE_FLOAT;
   mluOpDataType_t output_data_type = MLUOP_DTYPE_COMPLEX_FLOAT;
   mluOpDataType_t execution_dtype = MLUOP_DTYPE_FLOAT;
   const int rank = 1;
   const int batch = 2000;
   const int n[rank] = {400};
   const int ndim = rank + 1;
   const int input_dim_size[ndim] = {batch, n[0]};
   const int input_dim_stride[ndim] = {n[0], 1};

   const int output_dim_size[ndim] = {batch, n[0] / 2 + 1};
   const int output_dim_stride[ndim] = {n[0] / 2 + 1, 1};

   mluOpCreateTensorDescriptor(&input_desc);
   mluOpCreateTensorDescriptor(&output_desc);
   mluOpSetTensorDescriptorEx(input_desc, MLUOP_LAYOUT_ARRAY, input_data_type, ndim, input_dim_size, input_dim_stride);
   mluOpSetTensorDescriptorOnchipDataType(execution_dtype);
   mluOpSetTensorDescriptorEx(output_desc, MLUOP_LAYOUT_ARRAY, output_data_type, ndim,
                             output_dim_size, output_dim_stride);
   size_t reservespace_size;
   size_t workspace_size;
   mluOpMakeFFTPlanMany(handle, fft_plan, input_desc, output_desc, rank, n, &reservespace_size, &workspace_size);
   mluOpDestroyTensorDescriptor(input_desc);
   mluOpDestroyTensorDescriptor(output_desc);
   ```

3. 给plan绑定reservespace指针

   ```c
   void *reservespace;
   cnrtMalloc(&reservespace, reservespace_size);
   mluOpSetReserveArea(handle, fft_plan, reservespace);
   ```

4. 执行FFT，plan创建好以后可以执行多次

   ```c
   void *workspace;
   cnrtMalloc(&workspace, workspace_size);
   const float scale = 1.0;
   mluOpStatus_t mluOpExecFFT(handle, fft_plan, input, scale, workspace, output, 0);
   cnrtFree(workspace);
   ```pull/902/files#diff-7274399dd2d36c9d582d793971e1ecb6a43564088f8c10c16d32d6297520bd5b

5. 算子运行完以后释放plan，释放reservespace。

   ```c
   mluOpDestroyFFTPlan(fft_plan);
   cnrtFree(reservespace);
   ```

## 3 实现方案设计

参考文档：

www.pikara.ne.jp/okojisan/otfft-en/

https://users.ece.cmu.edu/~franzf/papers/fft-enc11.pdf

https://zh.wikipedia.org/wiki/克罗内克积

https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2008-62.pdf

www.fftw.org

https://docs.nvidia.com/cuda/cufft/index.html

### 3.1 实现方案

参照franz、cufft、OTFFT、和Microsoft的"Fast Computation of General Fourier Transforms on GPUs"的方案进行设计。

针对2的整数次幂采用StockhamFFT算法进行计算，质因数分解采用Six-Step FFT，对于非上述两种情况采用Blustein z-chirp方法进行计算。对于WRAM上面能直接放下Fourier Matrix的情境，可以考虑直接采用O(n^2)的暴力方法进行计算。

#### 3.1.1 Iterative FFT：Stockham FFT Algorithm

用来处理2的整数次幂的部分，Stockham FFT属于迭代式Cooley-tukey公式的变种，可以由Cooley Tukey FFT变换得到。长度为n的Cooley-tukey计算公式如下：
```math
\text{DFT}_n=(\text{DFT}_k \otimes I_m) T_m^n(I_k \otimes \text{DFT}_m) L_k^n,n=km,\tag{1}
```
其中$`\otimes`$表示为克罗内克乘积，$`(\text{DFT}_k \otimes I_m)`$为向量并行矩阵；$`T_m^n`$为旋转因子，为对角矩阵；$`(I_k \otimes \text{DFT}_m)`$为块内并行矩阵；$`L_k^n`$形式为$`L_m^{mn}:={in+j \rightarrow jm+i, 0\le i&lt;m, 0 \le j < n}`$ 的置换矩阵。下面展示了 $`k=4,m=2`$下的矩阵：
```math
I_4 \otimes \text{DFT}_2=\left[
\begin{array}{cccccccc}
1&1&&&&&&\\
1&1&&&&&&\\
&&1&1&&&&\\
&&1&-1&&&&\\
&&&&1&1&&\\
&&&&1&-1&&\\
&&&&&&1&1\\
&&&&&&1&-1\\
\end{array}
\right]

I_4 \otimes \text{DFT}_2=\left[
\begin{array}{cccccccc}
1&&&&1&&&\\
&1&&&&1&&\\
&&1&&&&1&\\
&&&1&&&&1\\
1&&&&-1&&&\\
&1&&&&-1&&\\
&&1&&&&-1&\\
&&&1&&&&-1\\
\end{array}
\right]
```
从上式可见，在满足$n=r^l$的条件下，$(1)$ 式可以不断展开写为迭代形式，得到迭代形式的 radix-r Cooley-tukey表达式：
```math
\text{DFT}_{r^l}=\left(\prod_{i=0}^{l-1}(I_{r^i}\otimes \text{DFT}_r\otimes I_{r^{l-i-1}})D_i^{r^l}\right)R_r^l,\tag{2}
```
其中，$D_i^{r^l}$ 为第$i$阶段的旋转因子矩阵，
```math
D_i^{r^l}=DiagMatrix[exp(\frac{2 \pi i}{r^l}r^i \alpha \beta), 0 \le \alpha < r, 0 \le \beta < r^{l-i}], \tag{3}
```
$R_r^{r^l}$为bit-reverse矩阵，即按位逆序置换矩阵。下面给出长度为8，基为2的置换矩阵表达形式：
```math
(x_0, x_4, x_2, x_6, x_1, x_5, x_3, x_7)^T=
\left(
\begin{array}{cccccccc}
1&&&&&&&\\
&&&&1&&&\\
&&1&&&&&\\
&&&&&&1&\\
&1&&&&&&\\
&&&&&1&&\\
&&&1&&&&\\
&&&&&&&1\\
\end{array}
\right) (x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7)^T.\tag{4}
```
可以看到$`R_r^l`$的离散取数逻辑对于硬件并不友好，经过对$`(2)`$式进行变换，尝试将$`R_r^l`$矩阵变换，可以得到Stockham FFT的形式：
```math
\text{DFT}_r^l=\prod_{i=0}^{l-1}(\text{DFT}_r\otimes I_{r^{l-1}})D_i^{r^l}(L_r^{r^{l-i}}\otimes I_{r^i})\tag{5}.
```
其中旋转因子矩阵为
```math
\text{DiagMatrix}\left[ W_{r^{j+2}}^{(\alpha r^j +\xi) \beta}, 0 \leq \alpha, \beta < r, 0 \leq \gamma < r^{l-j-2}, 0 \leq \xi < r^j \right] \tag{6}
```


注意这里的$`D_i^{r^l}$和$(2)`$中的旋转因子矩阵虽然具有相同的符号，但是其公式并不相同。可以看到$`(4)`$式具有固定的变换格式，并且左矩阵$`(\text{DFT}_r\otimes I_{r^{l-1}})`$具有向量并行的形式。适用于向量编程。我们使用Stockham FFT作为我们的迭代FFT的设计方案。

#### 3.1.2 Recursive FFT：Four-step FFT Algorithm

循环FFT是将$`n`$拆分为$`km`$的形式，式$`(1)`$就是一种Recursive FFT的计算形式，根据不同的硬件平台，我们可以将向量并行矩阵和块内并行矩阵进行转化来适应各种硬件平台。其中将块内并行矩阵转化的表达形式为4步法FFT：
```math
\text{DFT}_{n}=(\text{DFT}_{k}\otimes I_{m})T_{m}^{n} L_{k}^{n} (\text{DFT}_{m} \times I_{k}), n=km, \tag{7}
```
类似的，还有6步法FFT：
```math
\text{DFT}_n=L_k^n(I_m \otimes \text{DFT}_k) L_m^n T_m^n (I_k \otimes \text{DFT}_m) L_k^n, n=km, \tag{8}
```
以6步法FFT为例，OTFFT里面给出了很好的解释：

6步法FFT $`F_n`$可以看做是一个双重 $`G_n`$ ，代数表达方式如下：
```math
[k_1+k_2 m]=G_n(x[p_1+p_2 k])=\frac{1}{k}\sum_{p_1=0}^{k-1}\left(\left(\frac{1}{m}\sum_{p_2=0}^{m-1}x[p_1+p_2 k]W_m^{k_1 p_2}\right)W_n^{k_1 p_1}\right)W_k^{k_2 p_1} \tag{9}
```
可以看到，上式是FFT的组合，计算$`X`$ 的计算步骤如下：

​	Step 1. 转置$`x`$
```math
x[p_1+p_2 m]->a[p_2+p_1k] \tag{10}
```
​	Step 2. 对$`a`$ 中所有的 $`p_1`$ 分别做 $`F_m`$ 的FFT：
```math
a[p_2+p_1m]->b[k_1+p_1 m] \tag{11}
```
​	Step 3. 乘以旋转因子 $`W_n^{k_1 p_1}`$ ：
```math
b[k_1+p_1m]->b[k_1+p_1 m]W_n^{k_1 p_1}=c[k_1+p_1 m]\tag{12}
```
​	Step 4. 对 $`c`$ 进行转置：
```math
c[k_1+p_1 m]->d[p_1+k_1 k] \tag{13}
```
​	Step 5. 对所有的 $`k_1`$ 做 $`F_k`$ 的FFT：
```math
d[p_1+k_1 k]->e[k_2+k_1 k] \tag{14}
```
​	Step 6. 转置 $`e`$ ：
```math
e[k_2+k_1 k]->X[k_1+k_2 m] \tag{15}
```
其中，第2步和第6步的FFT可以使用3.1.1节中的迭代附列变换完成。

6步法FFT可以将长序列的FFT转化为一系列短序列的FFT，所以该方法对于长序列表现较好。

#### 3.1.3 General FFT：Bluestein chirp-z Algorithm

Bluestein z-chirp算法并不要求数据的长度具有合数的性质，将$`km`$写为$`km=(k^2+m^2-(k-m)^2)/2`$。
```math
\large X[k]=e^{-\frac{\pi i}{n} k^2}\sum_{j=0}^{n-1}\left(x_j e^{-\frac{\pi i}{n}j^2}\right)e^{\frac{\pi i}{n}(k-j)^2},0\le j < n \tag{16}
```
可见上式可以看做是带有 Scale 的两个向量的卷积形式：
```math
\large
\begin{align}
a_j&=x_j e^{-\frac{-\pi i}{n}j^2}\\
b_j&=e^{\frac{\pi i}{n}j^2}
\end{align} \tag{17}
```

```math
X_j=b_j^{*}\left(\sum_{j=0}^{n-1}a_j b_{k-j}\right), \tag{18}
```

$`a_j, b_j`$的分别卷积可以使用 zero-padding 到 $`r^l`$ 次方然后使用快速傅里叶变换进行实现，可以看到，该算法也具有 $`O(nlogn)`$ 的计算复杂度。但是由于3次傅里叶变换，该算法比 Cooley-Tukey 算法要慢。


#### 3.1.4 暴力解法：Direct FFT Algorithm

如果不考虑上述FFT的优化形式，我们可以直接将$`\text{DFT}_n`$直接写为Hermitian矩阵的形式：
```math
\text{DFT}_n= [W_n^{ij}, i\in[0,n),j\in[0,n)], \tag{19}
```
上述算法的计算复杂度为$O(n^2)$，但是对于一些具有矩阵计算单元的处理器来说，在小规模下性能不差于cooley-tukey FFT算法。

下面以 $`\text{DFT}_8`$ 为例进行展示：
```math
\text{DFT}_8= \left[
\begin{array}{cccccccc}
W_8^0&W_8^0&W_8^0&W_8^0&W_8^0&W_8^0&W_8^0&W_8^0&\\
W_8^0&W_8^1&W_8^2&W_8^3&W_8^4&W_8^5&W_8^6&W_8^7&\\
W_8^0&W_8^2&W_8^4&W_8^6&W_8^8&W_8^{10}&W_8^{12}&W_8^{14}&\\
W_8^0&W_8^3&W_8^6&W_8^9&W_8^{12}&W_8^{15}&W_8^{18}&W_8^{21}&\\
W_8^0&W_8^4&W_8^8&W_8^{12}&W_8^{16}&W_8^{20}&W_8^{24}&W_8^{28}&\\
W_8^0&W_8^5&W_8^{10}&W_8^{15}&W_8^{20}&W_8^{25}&W_8^{30}&W_8^{35}&\\
W_8^0&W_8^6&W_8^{12}&W_8^{18}&W_8^{24}&W_8^{30}&W_8^{36}&W_8^{42}&\\
W_8^0&W_8^7&W_8^{14}&W_8^{21}&W_8^{28}&W_8^{35}&W_8^{42}&W_8^{49}&\\
\end{array}
\right], \tag{20}
```
再次将 $\text{DFT}_8$矩阵的实部和虚部进行分行，再利用实数傅里叶分解的对称共轭特性，可以将 FFT 写为实数矩阵形式：
```math
\left[
\begin{array}{c}
ReX_0\\
ImX_0\\
ReX_1\\
ImX_1\\
ReX_2\\
ImX_2\\
ReX_3\\
ImX_3\\
ReX_4\\
ImX_4\\
\end{array}
\right]
=
\left[
\begin{array}{cccccccc}
Re(W_8^0&W_8^0&W_8^0&W_8^0&W_8^0&W_8^0&W_8^0&W_8^0)&\\
Im(W_8^0&W_8^0&W_8^0&W_8^0&W_8^0&W_8^0&W_8^0&W_8^0)&\\
Re(W_8^0&W_8^1&W_8^2&W_8^3&W_8^4&W_8^5&W_8^6&W_8^7)&\\
Im(W_8^0&W_8^1&W_8^2&W_8^3&W_8^4&W_8^5&W_8^6&W_8^7)&\\
Re(W_8^0&W_8^2&W_8^4&W_8^6&W_8^8&W_8^{10}&W_8^{12}&W_8^{14})&\\
Im(W_8^0&W_8^2&W_8^4&W_8^6&W_8^8&W_8^{10}&W_8^{12}&W_8^{14})&\\
Re(W_8^0&W_8^3&W_8^6&W_8^9&W_8^{12}&W_8^{15}&W_8^{18}&W_8^{21})&\\
Im(W_8^0&W_8^3&W_8^6&W_8^9&W_8^{12}&W_8^{15}&W_8^{18}&W_8^{21})&\\
Re(W_8^0&W_8^4&W_8^8&W_8^{12}&W_8^{16}&W_8^{20}&W_8^{24}&W_8^{28})&\\
Im(W_8^0&W_8^4&W_8^8&W_8^{12}&W_8^{16}&W_8^{20}&W_8^{24}&W_8^{28})&

\end{array}
\right]
\left[
\begin{array}{c}
x_0\\
x_1\\
x_2\\
x_3\\
x_4\\
x_5\\
x_6\\
x_7\\
\end{array}
\right], \tag{21}
```
之后可以根据矩阵乘设计方案进行设计，该方案一条指令即可完成一个短序列的FFT操作，并且对于长度大小没有限制。经过实测，X4计算卡处理 float32类型的 $`(2659, 400) * (402, 400)^T`$ 的耗时为 70us 左右，而T4下该规模的cufft 性能为 200us 左右。

在短时傅里叶变换(short-time fourier transformation, stft)的应用场景中，$n$ 的长度一般较短，该方案可以满足绝大多数的短时傅里叶变换的场景。

### 3.2调用方案设计

目前实现的方案如下：

- 针对GDRAM和NRAM的访存特点，设计片上片下 two-level 蝶形网络
  - 两层蝶形网络均采用stockham蝶形网络
  - 片下蝶形网络存放于GDRAM中， 片上网络存放于NRAM中
  - GDRAM根据片下蝶形网络执行顺序，负责将一个大基蝶形（large-radix butterfly）加载到NRAM上，在NRAM上将这个大基视为一个新的DFT，故再次进行FFT的分解，在片上蝶形网络中调用小基（small radix）kernel实现计算。
  - two-level的设计能够减少数据对于GDRAM的访存次数，例如对于n=[2048]的FFT，可以直接以一次大基radix-2048的形式执行（其可以完全存储在NRAM上），故只需向GDRAM进行一次load和一次store。
  - 举个例子，n=[8192]的FFT可以分解为大基为 256-32 的片下蝶形网络，在片上网络中，大基256可以继续分解为小基为16-16的蝶形网络，大基32直接分解为一个小基32。大基和小基的界定并非有明确的大小关系，例如大基可以为radix-4，而小基可以为radix-32，他们的区别是由哪个蝶形网络来调用。

- 蝶形计算目前采用matmul指令实现
  - 实现了generic的kernel实现蝶形计算，可以用于计算radix-2~64的小基

- 步骤如下
   - device端调用片下蝶形网络执行代码，c2c1d行主序对应函数为`computeMutiStageOnchip`，代码位于 kernels/fft/fft_optm_device/fft_two-level_network_c2c_device.mlu
   - computeMutiStageOnchip获取fft_plan中的网络参数，执行各个stage的大基实现，例如第一级大基对应函数`computeLargeButterflyFirststage`，代码位于 kernels/fft/fft_optm_device/fft_c2c_stockham_gdram.h
   - 大基执行函数亦为片上网络，与片下网络类似，继续调用小基kernel，代码位于 kernels/fft/fft_optm_device/fft_c2c_stockham_nram.h
   - 小基kernel代码分为为特定基实现的向量化版本（kernels/fft/fft_optm_device/fft_vector_butterfly.h）以及generic的矩阵实现版本（kernels/fft/fft_optm_device/fft_generic_butterfly.h）

### 3.3 拆分(任务拆分，多核拆分)

1. 一个ipu负责处理连续的多个batch。
2. mpu负责为一个cluster加载可共用的数据，如twiddles和DFT matrix。
### 3.5 方案理论性能

完成上述3.1，3.2，3.3，3.4几个步骤之后，基本可以给出一个理论性能，不需要每一个算子都有过于复杂的公式，但是一定要对自己的算子有一个心理的预期，最终实现之后的效率值是多少。

### 3.6 可维护性设计

1、bangc代码中加入必要的 log信息，比如输入的规模、数据类型、layout这些，以及如果出错会导致程序core dump的变量，比如IO指令的data_size、dim xyz的值等，这些信息都是有利于快速定位问题。

2、对每一个函数命名变量命名都有充分的注释

3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代   (待提供公共宏文档)

### 3.7 测试用例设计

- 框架在需求列表中给出的算子在网络中用到的规模：
  [2495, 400]

- 边界case：
  [1, 4096]

  [300, 1]

  [5, 0], n=[100]

  input stride

  output stride

  input stride + output stride

  input stride +inplace

  output stride + inplace

  input stride + output stride + inplace

其他可根据需要进行补充。算子开发完毕后，补充测试报告链接。

### 3.8 算子防呆检查


- 列出算子需要做的防呆，比如

 1、handle, plan, desc指针为空防呆；

 2、rank 为 1， 2， 3；

 3、输入输出维度防呆；

 4、输入输出数据类型防呆，针对r2c, c2r, c2c分别防呆；

 5、batch 大小防呆；

 6、execution dtype 数据类型防呆。

 7、输入输出stride防呆；

 8、signal length防呆，rfft：output = n / 2 + 1，irfft：output = n，fft： output = n；

 9、输入输出空指针防呆，如果输入元素不为0，防空指针；输出必不为空指针。

  9、 2-d，3-d fft，尚未支持。

 10、c2c, c2r fft防呆，尚未支持。

 11、r2c，n[0] > 4096，尚未支持。

