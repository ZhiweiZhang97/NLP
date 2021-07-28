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

Add & Norm模块接在Encoder和Decoder端每个子模块的后面，其中Add表示残差连接，Norm表示LayerNorm，因此Encoder端和Decoder端每个子模块实际的输出为: LayerNorm(x + Sublayer(x))，其中Sublayer(x)为子模块的输出.

### Positional Encoding

由于transformers中不具有循环和卷积结构，为了使模型能够利用序列的顺序信息，因此引入了Positional Encoding来解决这一问题.

$ PE_{(pos,2i)} = sin(pos/10000^{2i/d_{model}}) $

$ PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_{model}}) $

其中pos为位置，i为维度. 之所以这样定义position encoding，是希望position encoding满足下列的特性：
- 每个位置有一个唯一的positional encoding. 
- 两个位置之间的关系可以通过他们位置编码间的仿射变换来建模（获得），即任意位置$PE_{pos + k}$可以表示为$PE_{pos}$的线性函数.

Transformer中，Positional Encoding不是通过网络学习得来的，而是直接通过上述公式计算而来的，论文中也实验了利用网络学习Positional Encoding，发现结果与上述基本一致，但是**因为三角公式不受序列长度的限制，也就是可以对比所遇到序列的更长的序列进行表示**，因此选择了正弦和余弦函数版本.


