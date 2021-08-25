# NLP

## 评估标准

在模型评价的过程中，往往需要使用各种不同的指标进行评价。在众多评价指标中，大部分指标只能片面地反映模型的部分性能。如果评价指标不能合理使用，不仅模型本身找不到问题，还会得出错误的结论。定义TP、FP、FN、TN为true positive; false positive; false negative; true negative.

**Accuracy:** 准确性的定义是预测正确的结果占样本总数的百分比.
$
Accuracy = \frac{TP + TN}{TP + FP + FN + TN}
$

**Precision:** 表示在所有预测为真的样本中样本为真的概率，也就是说在预测为真的样本的结果中，预测正确的精度有多大.
$
Precision = \frac{TP}{TP + FP}
$

**Recall:** 在实际样本中被预测为正样本的概率.
$
Recall = \frac{TP}{TP + FN}
$

**Precision Recall Curve:** 描述查全率/查全率变化的曲线. 根据预测结果对测试样本进行排序，将最有可能是“正”的样本排在前面，最不可能是“正”的样本排在后面. 之后，按照这个顺序，将样本作为“正例”进行预测，并每次计算当前P值和R值.

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/PRC.png" width="300"/>

**F1-score:** P和r的调和平均值，F1越高，模型的性能越好.
$
F1 = \frac{2\times Precision\times Recall}{Precision+Recall}
$
对于多分类问题，估计全局性能的方法有宏观平均法(先对每一个类统计指标值，然后在对所有类求算术平均值)和微观平均法(对数据集中的每一个实例不分类别进行统计建立全局混淆矩阵，然后计算相应指标)两种.
$
P_{macro} = \frac{1}{n}\sum_{i=1}^n Precision_i, P_{micro} = \frac{\sum_{i=1}^n TP_i}{\sum_{i=1}^n (TP_i + FP_i)}
$

$
R_{macro} = \frac{1}{n}\sum_{i=1}^n Recall_i, R_{micro} = \frac{\sum_{i=1}^n TP_i}{\sum_{i=1}^n (TP_i + FN_i)}
$

$
F_{macro} = \frac{2 \times P_{macro} \times R_{macro}}{P_{macro} + R_{macro}}, F_{micro} = \frac{2 \times P_{micro} \times R_{micro}}{P_{micro} + R_{micro}}
$

**ROC曲线、AUC值:** 接收者操作特征曲线（receiver operating characteristic curve），是反映敏感性和特异性连续变量的综合指标，ROC曲线上每个点反映着对同一信号刺激的感受性. ROC曲线:

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/ROC.webp" width="400"/>

- 横坐标: 伪正类率(False positive rate，FPR，FPR=FP/(FP+TN))，预测为正但实际为负的样本占所有负例样本的比例;
- 纵坐标: 真正类率(True positive rate，TPR，TPR=TP/(TP+FN))，预测为正且实际为正的样本占所有正例样本的比例.

理想情况: TPR应接近1，FPR接近0，即图中的（0,1）点. ROC曲线越靠拢（0,1）点，越偏离45度对角线越好.

AUC(Area Under Curve)被定义为ROC曲线下的面积，显然这个面积的数值不会大于1. 又由于ROC曲线一般都处于y=x这条直线的上方，所以AUC的取值范围一般在0.5和1之间. 使用AUC值作为评价标准是因为很多时候ROC曲线并不能清晰的说明哪个分类器的效果更好，而作为一个数值，对应AUC更大的分类器效果更好. AUC值越大的分类器，正确率越高.

- AUC = 1，是完美分类器，采用这个预测模型时，存在至少一个阈值能得出完美预测(绝大多数预测的场合，不存在完美分类器);
- 0.5 < AUC < 1，优于随机猜测. 这个分类器(模型)妥善设定阈值的话，能有预测价值;
- AUC = 0.5，跟随机猜测一样，模型没有预测价值;
- AUC < 0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测.

## 常见优化器: SGD、BGD、MBGD、Momentum、Adagrad、RMSprop、Adam
梯度下降原理：
$
\theta^{n+1} = \theta^n + \eta \bigtriangledown \theta J(\theta)
$

### SGD(Stochastic Gradient Descent)

更新原则: 单条数据就可对参数进行一次更新.
- 优点: 参数更新速度快.
- 缺点: 由于每次参数更新时采用的数据量小, 造成梯度更新时震荡幅度大, 但大多数情况是向着梯度减小的方向.

### BGD(Batch Gradient Decent)

更新原则: 所有数据都参与梯度的每一次更新(**一个batch中每一个参数需要更新的梯度取均值作为更新值**).
- 优点: 由于每次参数更新时采用的数据量大, 所以梯度更新时比较平滑.
- 缺点: 由于每次参数更新时采用的数据量大, 参数更新速度慢.

### MBGD(Mini-Batch Gradient Decent)
更新原则: 只有所有数据的一部分进行参数的更新.
- 优点: 相比于SGD, 由于参与梯度更新的数据量大, 所以梯度更新时相对平滑; 相比于BGD, 参与梯度更新的数据量小, 参数更新速度更快一些.
- 缺点: 没有考虑到数据集的稀疏度和模型的训练时间对参数更新的影响.

### Momentum(解决SGD梯度下降时的震荡问题)
更新原则: 通过引入$\gamma^{v_n}$（$\gamma$一般取0.9左右）, 加速SGD并抑制震荡.(MBGD通过小批量数据来平滑梯度更新)

$
v_{n+1} = \gamma^{v_n} + \eta \theta J(\theta)
$

$
\theta^{n+1} = \theta^n - v_{n+1}
$
- 优点: 通过加入$\gamma^{v_n}$，使得梯度方向不变的维度上速度变快, 梯度方向改变的维度上更新速度变慢, 从而达到加快收敛并减小震荡的目的.
- 缺点: 梯度方向不变时, 参数更新速度会越来越快, 但是在梯度方向改变时 ,梯度更新速度不能及时减小导致适应性差.

### Adagrad(Adaptive gradient algorithm: 解决不能根据参数重要性对不同参数进行不同程度更新问题)
更新原则: 对低频的参数做较大的更新, 对高频的参数做较小的更新.
$
\theta_{t+1, i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii}}}g_{t,i}
$
- 优点: 减少了学习率的手动调节.
- 缺点: 分母会不断积累, 导致学习率会收缩并最终变得很小.

### Adadelta(解决Adagrad分母不断积累, 导致学习率收缩变得非常小的问题)
更新原则: 和Adagrad相比, 就是分母的$G_{t,ii}$换成了过去的梯度平方的衰减平均值(指数衰减平均值$\sqrt{E[g^2]_t} + \epsilon$). 这个分母相当于梯度的均方根, 所以简写为$RMS[g]_t$. 此外还将学习率$\theta$换成了$RMS[\Delta\theta]$.

$
E_{t} = \gamma E_{t-1} + (1 + \gamma)g^2_t
$

$
\theta_{t+1} = \theta_t + \Delta\theta_t = \theta_t - \frac{RMS[\Delta\theta]}{RMS[g]_t}g_t
$