# Transformers

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/Transformers.webp" width="400"/>

Transformers是一个典型的encoder-decoder模型.由6个encoder和6个decoder组成.
- **Encoder** 包含两层，一个多头self-attention层和一个前馈神经网络，self-attention能帮助当前节点不仅仅只关注当前的词，从而能获取到上下文的语义.
    - Encoder 端每个大模块接收的输入是不一样的，第一个大模块接收的输入是输入序列的embedding(embedding可以通过word2vec预训练得来)，其余大模块接收的是其前一个大模块的输出，最后一个模块的输出作为整个Encoder端的输出.
- **Decoder** 也包含encoder提到的两层网络，但是在这两层中间还有一层attention层（多头 Encoder-Decoder attention 交互模块），帮助当前节点获取到当前需要关注的重点内容.
    - Decoder第一个大模块训练时和测试时的接收的输入是不一样的，并且每次训练时接收的输入也可能是不一样的(也就是模型总览图示中的"shifted right")，其余大模块接收的是同样是其前一个大模块的输出，最后一个模块的输出作为整个Decoder端的输出.