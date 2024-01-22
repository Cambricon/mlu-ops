## MLU-OPS™ 算子精度验收标准

为了量化表示 MLU-OPS™ 算子与 CPU 或其他设备的精度差异， MLU-OPS™ 提出一套精度验收标准供开发者衡量算子质量。

### 1. 精度评价公式

MLU-OPS™ 正在使用的误差度量方式及其含义

|     | 误差度量指标 |       含义       |
| :-- | :----------: | :--------------: |
| 1   |    $diff1$     |   平均相对误差    |
| 2   |    $diff2$     |   标准相对误差    |
| 3   |    $diff3$     |   综合单点误差 |
| 4   |    $diff3_1$   |   最大单点相对误差 |
| 5   |   $diff3_2$    |   最大单点绝对误差 |
| 6   |    $diff4$     |   误差有偏性度量   |

各评价指标计算公式如下：

$evaluated\\_data$ 表示待评价数据， $baseline\\_data$ 表示基准数据。通常情况下， $baseline\\_data$ 为 CPU 计算结果或其他计算设备计算结果。

**$diff1$：平均相对误差**

$$
diff1 = \frac{ \sum |evaluated \\_ data - baseline \\_ data|}{\sum  |baseline \\_ data| } 
$$

**$diff2$：标准相对误差**

$$
diff2 = \sqrt{\frac{ \sum (evaluated \\_ data - baseline \\_ data)^2}{\sum  baseline \\_ data^2 }}
$$

**$diff3_1$：最大单点相对误差**

$$
diff3_1 = \mathop{max}\limits_{i}\frac{|evaluated \\_ data_i - baseline \\_ data_i|}{ |baseline \\_ data_i| }
$$

**$diff3_2$：最大单点绝对误差**

$$
diff3_2 = \mathop{max}\limits_{i}|evaluated \\_ data_i - baseline \\_ data_i|
$$

**$diff3$：综合单点误差**

$diff3 = (m_1, m_2)$ 。根据阈值 $th$ 将比较数据 $P$ 划分为两部分，对应下标 $i$ 的集合记作 $P_1$ 、 $P_2$ ，计算 $P_1$ 集合最大单点相对误差 $m_1$ 与 $P_2$ 集合最大单点绝对误差 $m_2$ 。 $P_1$ , $P_2$ , $m_1$, $m_2$ 计算方式如下：

$$
P_1 = i \in P \space : \space |baseline\\_data_i| \gt th
$$

$$
P_2 = P - P_1
$$

$$
m_1 = \mathop{max}\limits_{i \in P_1}\frac{|evaluated \\_ data_i - baseline \\_ data_i|}{ |baseline \\_ data_i| }
$$

$$
m_2 = \mathop{max}\limits_{i \in P_2}|evaluated \\_ data_i - baseline \\_ data_i|
$$

当数据类型为 $fp32$ 时， $th$ 为 $1e^{-6}$；当数据类型为 $fp16$ 时， $th$ 为 $1e^{-4}$ 。

**$diff4$：误差有偏性度量**

$diff4 = (p_1, p_2, n)$ ，计算方法为

$$
count_1 = \sum\limits_{ i } int(evaluated\\_data_i \gt baseline\\_data_i)
$$

$$
count_2 = \sum\limits_{ i } int(evaluated\\_data_i \lt baseline\\_data_i)
$$

$$
p_1 = \frac{count_1}{n}
$$

$$
p_2 = \frac{count_2}{n}
$$

$count_1$ 表示 $evaluated\\_data$ 大于 $baseline\\_data$ 的个数， $count_2$ 表示 $evaluated\\_data$ 小于 $baseline\\_data$ 的个数， $n$  表示不相等的总点数。

### 2. 算子精度分类

不同类型算子的误差度量方式和精度验收标准不同，我们把 MLU-OPS™ 算子根据使用的指令、计算的逻辑做了算子类型划分 :

|     | 算子类型    |                                   解释                              |            固定阈值                |
| :-: | :----:      | :-----------------------------------------------------------------: | :--------------------------------: |
|  1  | 卷积类      | 使用了卷积指令的算子 | $fp32: diff1 \leq 1e^{-5}$ && $diff2 \leq 1e^{-5}$<br />$fp16: diff1 \leq 3e^{-3}$ && $diff2 \leq 3e^{-3}$   |
|  2  | 累加类      | 规约求和类算子。比如 reducesum、reducemean。结果与累加顺序有关。    | $diff1 \leq 3e^{-3}$ && $diff2 \leq 3e^{-3}$     |
|  3  | 激活类      | 激活类算子是指算子的实现用到了线性插值、泰勒展开。                  | $diff1 \leq 3e^{-3}$ && $diff2 \leq 3e^{-3}$        |      
|  4  | 算术类      | 例如加减乘除。                                                      |             $diff3_2 == 0$              |
|  5  | 纯 IO       | 纯 IO 类的算子，不涉及任何运算。比如：concat、split。               |         $diff3_2 == 0$                  |
|  9  | 复合类      | 如果算子是由上面几种类型算子中的一种或几种复合组成，则算做复合算子。| $diff1 \leq 3e^{-3}$ && $diff2 \leq 3e^{-3}$      |
| 10  |atomicAdd 类 | 使用了 atomic 指令的算子，多次运行结果可能不同。                    | $diff1 \leq 3e^{-3}$ && $diff2 \leq 3e^{-3}$       |
