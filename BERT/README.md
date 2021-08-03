# BERT

BERT属于自编码语言模型(Autoencoder LM)，采用了**双向Transformer Encoder结构**，并且设计了两个任务来预训练模型.
- 第一个任务是采用MLM的方式来训练语言模型，通俗地说就是在输入一句话的时候，随机地选一些要预测的词，然后用一个特殊的符号[MASK]来代替它们，之后让模型根据所给的标签去学习这些地方该填的词. (**Masked LM**)
- 第二个任务在双向语言模型的基础上额外增加了一个句子级别的连续性预测任务，即预测输入BERT的两段文本是否为连续的文本，引入这个任务可以更好地让模型学到连续的文本片段之间的关系. (**Next Sentence Prediction**)

**与 Transformer 本身的 Encoder 端相比，BERT 的 Transformer Encoder 端输入的向量表示，多了 Segment Embeddings.** BERT相较于RNN、LSTM可以做到并发执行，同时提取词在句子中的关系特征，并且能在多个不同层次提取关系特征，进而更全面反映句子语义. 相较于 word2vec，能够根据句子上下文获取词义，从而避免歧义出现. 其缺点在于，模型参数太多，而且模型太大，少量数据训练时，容易过拟合.

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/BERT.jpeg" width="400"/>

对比单向Transformer结构的OpenAI GPT(AR)，BERT(AE)是双向的Transformer block连接，对比ELMo，虽然都是“双向”，但目标函数是不同的. ELMo分别以$P(w_i|w_1, ..., w_{i-1})$和$P(w_i|w_{i+1}, ..., w_n)$作为目标函数，独立处理两个representation然后拼接，而BERT则是以$P(w_i|w_1, ..., w_{i-1}, w_{i+1}, ..., w_n)$作为目标函数. ELMo尽管看上去利用了上文，也利用了下文，但是本质上仍然是自回归LM，这个跟模型具体怎么实现有关系. ELMO是做了两个方向(从左到右以及从右到左两个方向的自回归语言模型)，然后把LSTM的两个方向的隐节点状态拼接到一起，来体现双向语言模型这个事情的. 所以其实是两个自回归语言模型的拼接，本质上仍然是自回归语言模型。