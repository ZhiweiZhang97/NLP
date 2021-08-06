# GPT (Generative Pre-trained Transformer)

GPT的核心思想是先**通过无标签的文本去训练生成语言模型**(阶段一)，再根据具体的NLP任务**通过有标签的数据对模型进行fine-tuning**(阶段二). 具体来说，GPT结合了无监督的预训练和有监督的fine-tuning，并采用两阶段训练来适应下游任务.