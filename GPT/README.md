# GPT (Generative Pre-trained Transformer)

GPT的核心思想是先**通过无标签的文本去训练生成语言模型**(阶段一)，再根据具体的NLP任务**通过有标签的数据对模型进行fine-tuning**(阶段二). 具体来说，GPT结合了无监督的预训练和有监督的fine-tuning，并采用两阶段训练来适应下游任务.

## 模型结构

GPT采用多层双向Transformer结构，分为无监督训练和有监督fine-tuning两个阶段.

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/GPT.png" width="600"/>

GPT使用Transformer的Decoder结构，并对Transformer Decoder进行了一些改动，原本的Decoder包含了两个Multi-Head Attention结构，GPT只保留了Mask Multi-Head Attention.

### 1、无监督训练

给定无标签的文本$U = \lbrace{u_1, ..., u_n \rbrace}$，最大化语言模型的极大似然函数:
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

### 2、

