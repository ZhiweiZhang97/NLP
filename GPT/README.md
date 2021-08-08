# GPT (Generative Pre-trained Transformer)

GPT的核心思想是先**通过无标签的文本去训练生成语言模型**(阶段一)，再根据具体的NLP任务**通过有标签的数据对模型进行fine-tuning**(阶段二). 具体来说，GPT结合了无监督的预训练和有监督的fine-tuning，并采用两阶段训练来适应下游任务.

## 模型结构

GPT采用多层双向Transformer结构，分为无监督训练和有监督fine-tuning两个阶段.

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/GPT.png" width="600"/>

GPT使用Transformer的Decoder结构，并对Transformer Decoder进行了一些改动，原本的Decoder包含了两个Multi-Head Attention结构，GPT只保留了Mask Multi-Head Attention.

### 1、无监督训练

给定无标签的文本$U = \lbrace{u_1, ..., u_n \rbrace}$，最大化语言模型的极大似然函数(损失函数):
$
L_1(u) = \sum_{i}\log P(u_i|u_{i-k}, ..., u_{i-1}; \Theta)
$
其中，k是文本上下文窗口的大小.

input为词嵌入($W_e$)以及单词token的位置信息(W_p):
$
h_0 = UW_e + W_p
$

得到输入$h_0$之后，需要将$h_0$依次传入GPT的所有Transformer Decoder里，最终得到
$
h_t = transformer_block(h_{l-1}), l \in [1,t]
$

最后在预测下一个单词的概率
$
P(u) = softmax(h_tW_e^T)
$

### 2、有监督的fine-tuning

在对模型预训练之后，采用有监督的目标任务对模型参数微调. 给定一个有标签的数据集，假设每一条数据为一个单词序列$x_1, ..., x_m$，对应标签$y$. GPT微调的过程中根据单词序列$x_1, ..., x_m$预测标签$y$ .
$
P(y|x^1, ..., x^m) = softmax(h_l^mW_y)
$

损失函数为:
$
L_2(C) = \sum_{(x,y)} \log P(y|x^1, ..., x^m)
$

GPT 在微调的时候也考虑预训练的损失函数，所以最终的损失函数为:
$
L_3(C) = L_2(C) + \lambda * L_1(C)
$

### 具体任务微调

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/GPTT.png" width="600"/>

对于文本分类，只需要在预训练模型上微调。对于QA任务或者文本蕴含，因为预训练模型是在连续序列上训练，需要做一些调整，修改输入结构，将输入转化为有序序列输入.

1. **文本蕴含:** 将前提$p$和$h$序列拼接，中间用$(\$)$符号来分割两个序列.