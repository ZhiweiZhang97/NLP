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