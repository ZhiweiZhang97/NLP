# ELMo (Embeddings from Language Models)

ELMo采用了典型的两阶段过程: **1、利用语言模型进行预训练; 2、在做下游任务时，从预训练网络中提取对应单词的网络各层的Word Embedding作为新特征补充到下游任务中.**

### 1、利用语言模型进行预训练

ELMo采用了双层双向LSTM结构，其中单词特征采用的是单词的embedding或者采用字符卷积得到其embedding表示. 目前语言模型训练的任务目标是根据单词$W_i$的上下文(Context-before上文、Context-after下文)去正确预测单词$W_i$. 下图中，左端的前向双层LSTM代表正向编码器，输入的是从左到右的除了预测单词$W_i$外的上文; 右端的逆向双层LSTM代表反向编码器，输入的是从右到左的句子下文; 每个编码器的深度都是两层LSTM叠加. ELMo在层与层之间使用残差连接，以实现双向的效果. *ELMo尽管看上去利用了上文，也利用了下文，但是ELMo是做了两个方向(从左到右以及从右到左两个方向的自回归语言模型)，然后把LSTM的两个方向的隐节点状态拼接到一起，来体现双向语言模型这个事情的，所以其实是两个自回归语言模型的拼接，本质上仍然是自回归语言模型.*

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/ELMo1.jpeg" width="400"/>

利用大量语料对该网络结构进行训练之后，输入新句子$S_{new}$，句子中每个单词都能够得到对应的三个Embedding:
- 单词的Word Embedding(单词特征);
- 第一层LSTM中对应单词位置的Embedding(句法信息);
- 第二层LSTM中对应单词位置的Embedding(语义信息).

### 2、下游任务

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/ELMo2.png" width="400"/>

以QA问题为例: 对于问句X，先句子X输入到预训练好的ELMo网络中，得到句子X中每个单词对应的三个Embedding; 之后，分别给予这三个Embedding一个权重$\alpha_i$(这个权重可以通过学习得到)，根据各自权重进行累加求和，将三个Embedding整合成一个; 然后将整合后的这个Embedding作为X句在自己任务的那个网络结构中对应单词的输入，以此作为补充的新特征给下游任务使用. 对于QA问题的答句Y来说也做同样的处理. 因为ELMO给下游提供的是每个单词的特征形式，所以这一类预训练的方法被称为"Feature-based Pre-Training".

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/ELMo3.png" width="400"/>

## ELMo的训练过程

ELMo的训练过程实际上指的是其第一阶段的预训练过程(即训练一个双向语言模型). 假设给定一个含有N个Token的序列$(t_1, t_2, ..., t_N)$，那么:
- 前向语言模型在给定上文的情况下对$t_k$的概率建模来计算序列出现的概率: $p(t_1, t_2, ..., t_N) = \prod \limits_{k=1}^N p(t_k|t_1, t_2, ..., t_{k-1})$
    - 许多主流的神经语言模型都会先给序列中的Token计算一个**上下文无关**的Token表示$x_k^{LM}$，然后将它传递给L层前向LSTM. 这样的话，在每个位置k，每个LSTM层输出一个**上下文相关**的表示$\vec{h}_{k,j}^{LM}$，其中$j = 1, ..., L$(ELMo中取L=2).

- 后向语言模型与前向类似，但是它是"从后往前"的: $p(t_1, t_2, ..., t_N) = \prod \limits_{k=1}^N p(t_k|t_{k+1}, t_{k+2}, ..., t_N)$
    - 同样的，后向语言模型在每个位置k，每个LSTM层也输出一个**上下文相关**的表示$\overleftarrow{h}_{k,j}^{LM}$，其中$j = 1, ..., L$(ELMo中取L=2，$x_k^{LM}$是前后语言模型共享的).

综上，ELMo的训练过程即为一个前后向语言模型的训练过程，其训练目标就是最大化:
$
\sum_{k=1}^N(\log{p}(t_k|t_1, ..., t_{k-1};\Theta_x, \vec\Theta_{LSTM}, \Theta_s) + \log{p}(t_k|t_{k+1}, ..., t_N; \Theta_x, \overleftarrow\Theta_{LSTM}, \Theta_s))
$

其中， $\Theta_x$ 为Token表示的参数(前后向语言模型共享)，表示映射层的共享，表示第一步中，将单词映射为word embedding的共享，就是说同一个单词，映射为同一个word embedding; $\Theta_s$ 为softmax分类的参数(前后向语言模型共享)，表示第三步中的上下文矩阵的参数; $\vec\Theta_{LSTM}$, $\overleftarrow\Theta_{LSTM}$ 分别表示前向和后向语言模型LSTM层参数. ELMo的损失函数为简单的分类损失，取决于源码实现.

## How to use the trained ELMo？

对于序列中的每个Token，一个L层的双层语言模型会得到其**2L+1个表示**，即:
$
R_k = \lbrace{x_k^{LM}, \vec h_{k,j}^{LM}, \overleftarrow h_{k,j}^{LM}|j=1, ..., L \rbrace} = \lbrace{h_{k,j}^{LM}|j = 0, ..., L \rbrace}
$

其中，$h_{k,j}^{LM}$为Token表示(即$h_{k,0}^{LM} = x_k^{LM}$)，$h_{k,j}^{LM} = [\vec h_{k,j}^{LM}; \overleftarrow h_{k,j}^{LM}]$为每个双向LSTM层得到的表示. 这里是将整个句子输入到双向语言模型(双向LSTM网络)中，正向和反向LSTM网络共享Token Embedding的输入，源码中Token Embedding、正向、反向LSTM的hidden state均为512维度，一个长度为n的句子，经过ELMo预训练网络，最后得到的embedding的维度为: (n, 3, max_sentence_length, 1024). 在下游任务中:
$
ELMo_k^{task} = E(R_k; \Theta^{task}) = \gamma^{task}\sum_{j=0}^{L}s_j^{task}h_{k,j}^{LM}, s_j^{task} = e^{s_j}/\sum_{i}^{N}e^{s_i}
$

其中，$s^task$是经过soft Max归一化之后的权重，标量参数$\gamma^{task}$允许任务模型缩放整个ELMo向量($\gamma^{task}$是一个超参数，实际上这个参数是经验参数，一定程度上能够增强模型的灵活性). 综上，为下游任务获取Embedding的过程为:

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/ELMo4.png" width="400"/>

## ELMo的优缺点

### 优点:
- 引入双向语言模型，利用了深度上下文单词表征(由2个单向语言模型集成);
- 通过保存预训练好的2层biLSTM，通过特征集成或fine-tune应用于下游任务.

总结来说，通过上述结构，ELMo能够达到区分多义词的效果，每个单词(token)不再是只有一个上下文无关的Embedding表示.

### ELMo为什么有效:
- ELMo的假设前提是一个词的词向量不应该是固定的，所以在多义词区分方面ELMo的效果必然比word2vec要好;
- ELMo通过语言模型生成的词向量是通过特定上下文的“传递”而来，再根据下游任务，对原本上下文无关的词向量以及上下文相关的词向量表示引入一个权重，这样既在原来的词向量中引入了上下文的信息，又能根据下游任务适时调整各部分的权重(权重是在网络中学习得来的)，因此这也是ELMo有效的一个原因.

### 缺点:
- 使用LSTM提取特征，而LSTM提取特征的能力弱于Transformer;
- 使用向量拼接方式融合上下文特征，这种方式获取的上下文信息效果不如想象中好;
- 训练时间长，这也是RNN的本质导致的.

## ELMo为什么能够达到区分多义词的效果

ELMo是双层双向LSTM结构，每个单词对应两边LSTM网络的对应节点，在将句子输入模型中提取各层Embedding的时候，两个节点得到的Embedding是动态改变的，会受到上下文单词的影响，通过单词上下文的不同来强化当前语义，弱化其它语义，以达到区分多义词的效果.

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/ELMo5.webp" width="400"/>

## ELMo叠加三种不同向量的意义
1. 之前的很多方法都只用了最顶层LSTM的hidden state，但是通过实验验证，在很多时候，将每一层hidden state融合在一起会取得更好的效果;
2. 由于每一层LSTM得到的单词Embedding所蕴含的信息是不一样的，因此将所有的信息融合在一起，能够让单词Embedding的表达更丰富.



