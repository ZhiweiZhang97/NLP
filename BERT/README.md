# BERT

BERT属于自编码语言模型(Autoencoder LM)，采用了**双向Transformer Encoder结构**，并且设计了两个任务来预训练模型.
- 第一个任务是采用MLM的方式来训练语言模型，通俗地说就是在输入一句话的时候，随机地选一些要预测的词，然后用一个特殊的符号[MASK]来代替它们，之后让模型根据所给的标签去学习这些地方该填的词. (**Masked LM**)
- 第二个任务在双向语言模型的基础上额外增加了一个句子级别的连续性预测任务，即预测输入BERT的两段文本是否为连续的文本，引入这个任务可以更好地让模型学到连续的文本片段之间的关系. (**Next Sentence Prediction**)

**与 Transformer 本身的 Encoder 端相比，BERT 的 Transformer Encoder 端输入的向量表示，多了 Segment Embeddings.** BERT相较于RNN、LSTM可以做到并发执行，同时提取词在句子中的关系特征，并且能在多个不同层次提取关系特征，进而更全面反映句子语义. 相较于 word2vec，能够根据句子上下文获取词义，从而避免歧义出现. 其缺点在于，模型参数太多，而且模型太大，少量数据训练时，容易过拟合.

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/BERT.jpeg" width="400"/>

对比单向Transformer结构的OpenAI GPT(AR)，BERT(AE)是双向的Transformer block连接，对比ELMo，虽然都是“双向”，但目标函数是不同的. ELMo分别以$P(w_i|w_1, ..., w_{i-1})$和$P(w_i|w_{i+1}, ..., w_n)$作为目标函数，独立处理两个representation然后拼接，而BERT则是以$P(w_i|w_1, ..., w_{i-1}, w_{i+1}, ..., w_n)$作为目标函数. ELMo尽管看上去利用了上文，也利用了下文，但是本质上仍然是自回归LM，这个跟模型具体怎么实现有关系. ELMO是做了两个方向(从左到右以及从右到左两个方向的自回归语言模型)，然后把LSTM的两个方向的隐节点状态拼接到一起，来体现双向语言模型这个事情的. 所以其实是两个自回归语言模型的拼接，本质上仍然是自回归语言模型.

### 训练过程
#### 1、Masked LM

任务描述: 给定一句话，随机抹去这句话中的一个或几个词，要求根据剩余词汇预测被抹去的几个词分别是什么.

该任务类似于**完形填空**任务，训练过程中作者随机选择15%的词汇进行预测. 对于这15%的词汇，其中80%采用一个特殊符号[MASK]进行替换，10%采用一个任意词替换，剩余10%保持原词汇不变.

- **Why**: 
    - 在后续微调任务中语句中并不会出现[MASK]标记，且在预测一个词汇时，模型并不知道输入对应位置的词汇是否为正确的词汇(10% 概率)，这就迫使模型更多地依赖于上下文信息去预测词汇，并且赋予了模型一定的纠错能力. 但由于每批次数据中只有15%的标记被预测，这意味着模型可能需要更多的预训练步骤来收敛. 且由于Pre-training和Fine-tuning两个阶段存在使用模式不一致的情况(**在后续微调任务中语句中并不会出现[MASK]标记**)，可能会给模型带来一定的损失. 同时，被MASK标记的单词之间没有任何关系，是条件独立的，而有时这些单词之间是有关系的.

#### 2、Next Sentence Prediction

任务描述: 给定一篇文章中的两句话，判断第二句话在文本中是否紧跟在第一句话之后.

该任务实际上就是段落重排序的简化版: 只考虑两句话，判断是否是一篇文章中的前后句. 在实际预训练过程中，文章作者从文本语料库中随机选择50%正确语句对和50%错误语句对进行训练. 与Masked LM任务相结合，让模型能够更准确地刻画语句乃至篇章层面的语义信息，使模型输出的每个字/词的向量表示都能尽可能全面、准确地刻画输入文本的整体信息，为后续的微调任务提供更好的模型参数初始值.

### BERT VS. ELMo

从网络结构以及最后的实验效果来看，BERT比ELMo效果好的主要原因如下:
- LSTM抽取特征的能力远弱于Transformer;
- ELMO采取双向拼接这种融合特征的能力可能比Bert一体化的融合特征方式弱;
- BERT的训练数据以及模型参数均多余ELMo.

**区别:** ELMo模型是通过语言模型任务得到句子中单词的embedding表示，以此作为补充的新特征给下游任务使用. 因为ELMo给下游提供的是每个单词的特征形式，所以这一类预训练的方法被称为“Feature-based Pre-Training”. 而BERT是基于Pre-train Fine-tune方式，使用Pre-train参数初始化Fine-tune的模型结构.(这种做法和图像领域基于Fine-tuning的方式基本一致，下游任务需要将模型改造成BERT模型，才可利用BERT模型预训练好的参数.)

### BERT的局限性

1. BERT在Pre-training阶段MASK的多个单词之间没有任何关系，是条件独立的，然而有时候这些单词之间是有关系的
    - 比如”New York is a city”，假设Mask住”New”和”York”两个词，那么给定”is a city”的条件下”New”和”York”并不独立，因为”New York”是一个实体，看到”New”则后面出现”York”的概率要比看到”Old”后面出现”York”概率要大得多. (能够依靠海量的训练数据来减少这个问题对结果的影响，当训练语料足够大的时候总有例子能够学会这些单词的相互依赖关系.)
2. 由于在BERT的Pre-training阶段的特殊[MASK]标记不会出现在Fine-tuning阶段，这就导致了两阶段不一致的问题. 所以BERT的Masked LM任务中随机选择的15%的单词并没有被全部MASK掉，一定程度上缓解了这一问题.
3. 为了解决OOV问题，我们通常会把词分成更细粒度的WordPiece. BERT在Pre-training的时候是随机Mask这些WordPiece的，这就可能出现只 Mask一个词的一部分的情况. 对于该问题后续推出了WWM版本(Whole Word Masking).

4. BERT对生成式任务处理效果不好(自回归语言模型Pre-train模式的原因). BERT对超长文本效果不理想.

### BERT的输入输出

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/BERT_INPUT.png" width="400"/>

**输入:** 
1. 文本中各个字/词(或者称为token)的原始**词向量(Token Embeddings)**，该向量既可以随机初始化，也可以利用Word2Vector等算法进行预训练以作为初始值;
2. **文本向量(Segment Embeddings)**，该向量的取值在模型训练过程中自动学习，用于刻画文本的全局语义信息，并与单字/词的语义信息相融合;
3. **位置向量(Position Embeddings)**，由于出现在文本不同位置的字/词所携带的语义信息存在差异，因此，BERT模型对不同位置的字/词分别附加一个不同的向量以作区分.

**输出:** 输出是文本中各个字/词融合了全文语义信息后的向量表示.(全局语义表征向量 & 各个词的表征向量)

在做Next Sentence Prediction任务时，在第一个句子的首部会加上一个[CLS] token，在两个句子中间以及最后一个句子的尾部会加上一个[SEP] token.

### How to do Fine-tuning

1. 文本分类任务(How to Fine-Tune BERT for Text Classification?)

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/ftc.png" width="400"/>

**微调策略:** 采用多种方式在目标任务上微调BERT显然是非常有必要的. BERT不同的层可以捕获到不同级别的语法及语义信息，哪些信息是目标任务所需要的呢？如何选择优化算法以及学习率呢？当把BERT适应到某个目标任务时，需要考虑到因素有: 
- 长文本序列的预处理(BERT最大序列长度为512); 
    - 截断法(truncation methods): (1)head-only: 只保留前510个tokens; (2)tail-only: 只保留尾510个tokens; (3) **head+tail(表现最好)**: 根据经验选择前128个tokens与后382个tokens。
    - 层次法(hierarchical methods): 输入文本首先被分割成$k = L / 510$个切片，然后输入到BERT中得到$k$个切片的表征. 每个切片的表征就是最后一层的符号[CLS]的隐藏层状态. 可以使用 mean pooling、max pooling与self-attention的方式把所有的切片的表征合并起来。
- 选择哪些BERT层; 
- 过拟合问题.

2. 语义相似度任务

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/语义.png" width="400"/>

在实际操作时，最后一句话之后还会加一个[SEP] token，语义相似度任务将两个句子按照图中方式输入即可，之后与论文中的分类任务一样，将[CLS] token位置对应的输出，接上softmax做分类即可.

3. 多标签分类任务

利用 BERT 模型解决多标签分类问题时，其输入与普通单标签分类问题一致，得到其embedding表示之后(也就是BERT输出层的embedding)，有几个label就连接到几个全连接层(也可以称为projection layer)，然后再分别接上softmax分类层，这样的话会得到多个loss ，最后再将所有的loss相加起来即可. (相当于将n个分类模型的特征提取层参数共享，得到一个共享的表示(其维度可以视任务而定，由于是多标签分类任务，因此其维度可以适当增大一些)，最后再做多标签分类任务.)


