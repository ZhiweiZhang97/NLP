# GPT (Generative Pre-trained Transformer)

GPT的核心思想是先**通过无标签的文本去训练生成语言模型**(阶段一)，再根据具体的NLP任务**通过有标签的数据对模型进行fine-tuning**(阶段二). 具体来说，GPT结合了无监督的预训练和有监督的fine-tuning，并采用两阶段训练来适应下游任务.

## 模型结构

GPT采用多层双向Transformer结构，分为无监督训练和有监督fine-tuning两个阶段.

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/GPT.png" width="400"/>

GPT使用Transformer的Decoder结构，并对Transformer Decoder进行了一些改动，原本的Decoder包含了两个Multi-Head Attention结构，GPT只保留了Mask Multi-Head Attention.