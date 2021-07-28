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
- **Decoder** 也包含encoder提到的两层网络，但是在这两层中间还有一层attention层（多头 Encoder-Decoder attention 交互模块），帮助当前节点获取到当前需要关注的重点内容.
    - Decoder第一个大模块训练时和测试时的接收的输入是不一样的，并且每次训练时接收的输入也可能是不一样的(也就是模型总览图示中的"shifted right")，其余大模块接收的是同样是其前一个大模块的输出，最后一个模块的输出作为整个Decoder端的输出.
    - 对于第一个大模块，其训练及测试时接收的输入为：训练的时候每次的输入为上次的输入加上输入序列向后移一位的ground truth(例如每向后移一位就是一个新的单词，那么则加上其对应的embedding，通过使用多头attention模块对序列进行mask实现)，特别地，当decoder的time step为1时(第一次接收输入)，其输入为一个特殊的 token，其目标则是预测下一个位置的单词(token)是什么，对应到time step为1时，则是预测目标序列的第一个单词(token)是什么，以此类推；而在测试的时候，首先，生成第一个位置的输出，之后，在第二次预测时，将其加入输入序列，以此类推直至预测结束.（训练有真实数据，也就是提到的ground truth，因此通过mask操作来完成self attention，但是在inference的时候，没有ground truth，所以需要拿y^_t-1作为当前的输入，就变成了LSTM那种逐步解码的过程）
## Attention模块

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/selfA.webp" width="400"/>

Scaled Dot-Product Attention: **将query和key-value键值对的一组集合映射到输出**，其中query和keys的维度均为d_k，values的维度为d_v，输出被计算为values的加权和，其中每个value的权重由query和key的相似性函数计算得到，对应公式为：
<img align="center" src="http://chart.googleapis.com/chart?cht=tx&chl= Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V" style="border:none;">
