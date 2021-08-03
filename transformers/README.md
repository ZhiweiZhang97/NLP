<!-- <script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script> -->

# Transformers

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/Transformers.webp" width="400"/>

Transformers是一个典型的encoder-decoder模型.由6个encoder和6个decoder组成.
- **Encoder** 包含两层，一个多头self-attention层和一个前馈神经网络，self-attention能帮助当前节点不仅仅只关注当前的词，从而能获取到上下文的语义.
    - Encoder 端每个大模块接收的输入是不一样的，第一个大模块接收的输入是输入序列的embedding(embedding可以通过word2vec预训练得来)，其余大模块接收的是其前一个大模块的输出，最后一个模块的输出作为整个Encoder端的输出.
- **Decoder** 也包含encoder提到的两层网络，但是在这两层中间还有一层attention层（多头Encoder-Decoder attention交互模块），帮助当前节点获取到当前需要关注的重点内容.
    - Decoder第一个大模块训练时和测试时的接收的输入是不一样的，并且每次训练时接收的输入也可能是不一样的("shifted right")，其余大模块接收的是同样是其前一个大模块的输出，最后一个模块的输出作为整个Decoder端的输出.
    - 对于第一个大模块，其训练及测试时接收的输入为：训练的时候每次的输入为上次的输入加上输入序列向后移一位的ground truth(例如每向后移一位就是一个新的单词，那么则加上其对应的embedding，通过使用多头attention模块对序列进行mask实现)，特别地，当decoder的time step为1时(第一次接收输入)，其输入为一个特殊的 token，其目标则是预测下一个位置的单词(token)是什么，对应到time step为1时，则是预测目标序列的第一个单词(token)是什么，以此类推；而在测试的时候，首先，生成第一个位置的输出，之后，在第二次预测时，将其加入输入序列，以此类推直至预测结束.（训练有真实数据，也就是提到的ground truth，因此通过mask操作来完成self attention，但是在inference的时候，没有ground truth，所以需要拿$\hat{y}_{t-1}$作为当前的输入，就变成了LSTM那种逐步解码的过程）

### Attention模块

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/selfA.webp" width="400"/>

**Scaled Dot-Product Attention:** 将query和key-value键值对的一组集合映射到输出，其中query和keys的维度均为d_k，values的维度为d_v，输出被计算为values的加权和，其中每个value的权重由query和key的相似性函数计算得到.
$
    Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$

**Multi-Head Attention:** 将Q,K,V通过参数矩阵映射后(给Q,K,V分别接一个全连接层)，然后再做self-attention，将这个过程重复h次，最后再将所有的结果拼接起来，再送入一个全连接层.

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/mutiA.webp" width="400"/>

### 前馈神经网络

前馈神经网络模块(即图示中的 Feed Forward)由两个线性变换组成，中间有一个 ReLU 激活函数.
$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$

### 多头Encoder-Decoder attention交互模块

多头Encoder-Decoder attention交互模块的形式与Multi-Head Attention模块一致，唯一不同的是Q,K,V矩阵的来源，Q矩阵源于下方子模块的输出(对应模型图中masked Multi-Head Attention模块经过Add & Norm后的输出)，而K,V矩阵则源于整个Encoder端的输出. 这里的交互模块类似于seq2seq with attention中的机制，目的在于让Decoder端的单词(token)给予Encoder端对应的单词(token)“更多的关注(attention weight)”.

### Add & Norm模块

Add & Norm模块接在Encoder和Decoder端每个子模块的后面，其中Add表示残差连接，Norm表示LayerNorm，因此Encoder端和Decoder端每个子模块实际的输出为: LayerNorm(x + Sublayer(x))，其中Sublayer(x)为子模块的输出. (防止梯度消失，帮助深层网络训练).

### Positional Encoding

Transformer模型中缺少一种解释输入序列中单词顺序的方法，它跟序列模型还不不一样. 为了处理这个问题，transformer给encoder层和decoder层的输入添加了一个额外的向量Positional Encoding，维度和embedding的维度一样，这个向量采用了一种很独特的方法来让模型学习到这个值，这个向量能决定当前词的位置，或者说在一个句子中不同的词之间的距离.(由于transformers中不具有循环和卷积结构，为了使模型能够利用序列的顺序信息，因此引入了Positional Encoding来解决这一问题.)

$ PE_{(pos,2i)} = sin(pos/10000^{2i/d_{model}}) $

$ PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_{model}}) $

其中pos为位置，i为维度. 之所以这样定义position encoding，是希望position encoding满足下列的特性：
- 每个位置有一个唯一的positional encoding. 
- 两个位置之间的关系可以通过他们位置编码间的仿射变换来建模（获得），即任意位置$PE_{pos + k}$可以表示为$PE_{pos}$的线性函数.

Transformer中，Positional Encoding不是通过网络学习得来的，而是直接通过上述公式计算而来的，论文中也实验了利用网络学习Positional Encoding，发现结果与上述基本一致，但是**因为三角公式不受序列长度的限制，也就是可以对比所遇到序列的更长的序列进行表示**，因此选择了正弦和余弦函数版本.

### Some questions

#### Why Multi-head Attention

原论文中说进行Multi-head Attention的原因是将模型分为多个头，形成多个子空间，可以让模型去关注不同方面的信息，最后再将各个方面的信息综合起来. 直观上来说，多头注意力**有助于网络捕获到更丰富的特征/信息**，可以类比CNN中同时使用多个卷积核的作用.

在不增加时间复杂度的情况下，同时，借鉴CNN多核的思想，在更低的维度，在多个独立的特征空间，更容易学习到更丰富的特征信息.(transformer中multi-head attention中每个head为什么要进行降维？)

#### self-attention为什么要使用 Q、K、V，仅仅使用 Q、V/K、V或者V为什么不行

首先，使用不相同但Q、K、V可以保证模型在不同空间进行投影，增强了表达能力，提高了泛化能力. 如果令Q和K相同，那么得到的模型大概率会得到一个类似单位矩阵的attention矩阵，这样self-attention就退化成一个point-wise线性映射，对于注意力上的表现力不够强.

#### Self-Attention为什么能发挥如此大的作用

self-attention是一种通过自身和自身相关联，从而得到一个更好的representation来表达自身的Attention机制，可以看出是一般attention的一种特殊情况，self-attention中Q=K=V，序列中的每一个单词(token)和该序列中的其它单词(token)进行Attention计算. **self-attention无视词(token)之间的距离直接计算依赖关系，从而能够学习到序列的内部结构.** 在多数情况下，会对下游任务有一定的促进作用.

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/selfa1.jpeg" width="400"/>
<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/selfa2.jpeg" width="400"/>

从图中可以看出，self-attention可以捕获同一个句子中单词之间的一些句法特征或语义特征. 相对于RNN或LSTM的依次序序列计算，self-attention在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接的联系起来，极大的缩短了远距离依赖特征之间的距离，能够更容易的捕获句子中长距离的相互依赖的特征，并更加有效的利用这些特征. 同时self-attention对于增加计算的并行性也有直接的帮助.

self-attention计算方式: 将query和key-value键值对的一组集合映射到输出，其中query和keys的维度均为d_k，values的维度为d_v，输出被计算为values的加权和，其中每个value的权重由query和key的相似性函数计算得到.
$
    Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$

#### self-attention中归一化的作用

随着模型$d_k$的增大，$q \cdot k$的点积结果也会随之增大，这样会将softmax函数推入梯度非常小的区域，使得收敛困难(可能出现梯度消失的情况). 

假设向量$q$和$k$的各个分量是互相独立的随机变量，均值是0，方差是1，那么点积$q \cdot k$的均值是0，方差是$d_k$. 方差越大也就说明，点积的数量级越大(以越大的概率取大值). 那么一个自然的做法就是把方差稳定到1，做法是将点积除以$\sqrt{d_k}$，这样有$D(\frac{q \cdot k}{\sqrt{d_k}}) = \frac{d_k}{\sqrt{d_K}^2} = 1$. 将方差控制为1，也就**有效地控制了梯度消失的问题**.

#### Transformer相比于RNN/LSTM的优势

- RNN系列的模型并行能力很差
    - RNN系列模型T时刻隐层状态的计算依赖T时刻的句子输入单词$X_t$和T-1时刻的隐层状态的输出$S_{t-1}$. 由于RNN模型的当前时刻的计算依赖前一时刻的隐层状态的计算结果，从而形成了序列依赖关系，导致RNN系列模型的并行能力很差.

- Transformers的特征抽取能力比RNN系列模型更好

虽然Transformer在大部分情况下优于RNN/LSTM，但并不是说Transformer能够完全替代RNN/LSTM，任何模型都有其适用范围.

#### Transformer训练过程

Transformer的训练过程与seq2seq类似，Encoder端得到输入的encodeing表示，并将其输入到Decoder端做交互式Attention. Decoder端接收其相应的输入，经过Multi-head Attention之后，结合Encoder端的输出共同输入到Decoder端的交互式Attention模块，再经过FFN得到Decoder端的输出. 最后经过一个线性全连接层，Transformer可以通过softmax来预测下一个单词. **Encoder端可以并行计算，一次性得到输入序列的全部encodeing表示，但Decoder端并不是一次性得到所有的预测单词，而是类似seq2seq一个接一个的预测出来.**

#### Transformer & seq2seq

传统的seq2seq最大的问题在于**将 Encoder 端的所有信息压缩到一个固定长度的向量中**，并将其作为Decoder端首个隐藏状态的输入，来预测Decoder端第一个单词(token)的隐藏状态. 在输入序列比较长的时候，这样做显然会损失Encoder端的很多信息，而且这样一股脑的把该固定向量送入Decoder端，Decoder端不能够关注到其想要关注的信息. 后序的论文虽然对传统seq2seq的两个缺点有所改进，但由于主题模型仍为RNN系列的模型，因此模型的并行能力还是受限. 相较而言，Transformer中的**多头交互式Attention模块**对传统seq2seq模型的两个缺点对seq2seq模型的两点缺点有了实质性的改进，而self-attention模块通过自身和自身的关联，增强了自身的表达度，使的Transformer的表达能力更强，并且Transformer并行计算的能力远远超过seq2seq系列的模型.

#### Transformer中的Encoder表示

Transformer Encoder端得到的是整个输入序列的encoding表示，其中最重要的是经过了self-attention模块，让输入序列的表达更加丰富，而加入词序信息是使用不同频率的正、余弦函数.

#### Transformer如何并行化

Transformer的Decoder端无法并行计算，只能一个接一个的解码，类似RNN，当前时刻的输入依赖于上一个时刻的输出. 对于Encoder端，首先，6个大的模块之间是串行的，一个模块计算的结果做为下一个模块的输入，模块之间有依赖关系，而每个模块自身能够并行处理整个序列. (例如Encoder端的self-attention模块，对于某个序列$x_1, x_2, ..., x_n$，self-attention模块可以直接计算$x_i, x_j$的点乘结果.)