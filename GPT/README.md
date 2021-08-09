# GPT (Generative Pre-trained Transformer)

GPT的核心思想是先**通过无标签的文本去训练生成语言模型**(阶段一)，再根据具体的NLP任务**通过有标签的数据对模型进行fine-tuning**(阶段二). 具体来说，GPT结合了无监督的预训练和有监督的fine-tuning，并采用两阶段训练来适应下游任务. *GPT的每一层都保留了对前一个单词的解释，但不会更具下一个单词重新解释前一个单词.*

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

input为词嵌入($W_e$)以及单词token的位置信息(W_p. 与Transformer不同, GPT的位置编码并非是通过三角函数计算来的, 而是通过训练学习到的.):
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

1. **文本蕴含:** 将前提$p$和序列$h$拼接，中间用($)符号来分割两个序列.
2. **文本相似度:** 分别输入两个序列，通过GPT得到两个序列的特征向量，再逐元素相加输入线性层.
3. **问答和常识推理:** 给定上下文z，问题q，以及一组可能的候选答案{$a_k$}，将三者拼接起来，的到输入序列[z; q; $; $a_k$]，输入预训练模型，经过softmax层得到候选答案的概率分布.

### GPT1 / GPT2

- GPT1
    - 采用Transformer进行特征抽取，可以捕捉到更长范围的信息，首次将Transformer用于预训练语言模型;
    - fine-tune阶段引入语言模型辅助目标，解决fine-tune过程中的灾难性疑问;
    - 预训练和fine-tune一致，统一两阶段框架.
- GPT2
    - 没有针对特定模型的精调流程: GPT2认为预训练中已包含很多特定任务所需的信息;
    - 生成任务取得很好的效果，使用覆盖广、质量高的数据.
- 缺点
    - 依然为单项自回归语言模型，无法获取上下文相关的特征表示.
- Difference Between GPT1 and GPT2
    - 使用了更大的数据集;
    - 增加了海量参数，并推出了几个不同的版本(一个比一个大)，除了使用的DEcoder个数不同，它们的Embedding维度也是不同的;
    - 去掉了fine-tune层，直接输入任务和所需内容就能得到输出;
    - 将Layer Norm放到了每个子层的输入前，并且在最后一个自注意力层后添加了Layer Norm. 通过放缩权重更换了残差层的初始化方式.

