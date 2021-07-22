# CNN
·卷积神经网络的思想是捕捉局部特征，对于文本来说，局部特征就是由若干个单词组成的滑动窗口，类似于N-gram。

·CNN能够自动对N-gram特征进行组合和筛选，获得不同抽象层次的语义信息。每次卷积中采用了共享权重的机制，训练速度会更快。

<img src="https://user-images.githubusercontent.com/30019518/113801798-8529df80-9794-11eb-825f-ccb75b9eb335.png" width="400"/>

CNN通常由一个或多个卷积层（Convolutional Layer）和全连接层（Fully Connected Layer，对应经典的 NN）组成，也会包括池化层（Pooling Layer）
## Pooling Layer(max polling / average polling)：
- 1、在保留主要特征的同时减少参数和计算量，防止过拟合，提高模型的泛化性；
- 2、获得空间变换不变性（平移、旋转、缩放）

# RNN
循环神经网络的神经元除了接受前一状态的输出作为输入，还有自身的状态信息，其状态信息在网络中循环传递；RNN 把所处理的数据序列视作时间序列，在每一个时刻 t，每个 RNN 的神经元接受两个输入：当前时刻的输入样本 xt，和上一时刻自身的输出 ht-1. ht = F(ht-1, xt).

<img src="https://user-images.githubusercontent.com/RNN.jpeg" width="400"/>

# LSTM
每个神经元接受的输入除了当前时刻样本输入，上一个时刻的输出，还有一个元胞状态（Cell State）。如果把 LSTM 的遗忘门强行置0，输入门置1，输出门置1，则 LSTM 就变成了标准 RNN。

- 遗忘门：接受xt 和 ht-1 为输入，输出一个0到11之间的值，用于决定在多大程度上保留上一个时刻的元胞状态ct-1，1表示全保留，0表示全放弃；
- 输入门: 用于决定将哪些信息存储在这个时刻的元胞状态 ct 中；
- 输出门：用于决定输出哪些信息。

LSTM 在很大程度上缓解了一个在 RNN 训练中非常突出的问题：梯度消失/爆炸（Gradient Vanishing/Exploding）。因为三个门，尤其是遗忘门的存在，LSTM 在训练时能够控制梯度的收敛性，从而梯度消失/爆炸的问题得以缓解，同时也能够保持长期的记忆性。

<img src="https://user-images.githubusercontent.com/LSTM.png" width="400"/>

# 梯度消失/爆炸

梯度消失和梯度爆炸是困扰RNN模型训练的关键原因之一，产生梯度消失和梯度爆炸是由于RNN的权值矩阵循环相乘导致的，相同函数的多次组合会导致极端的非线性行为。

- 如果梯度小于1，那么随着层数增多，梯度更新信息将会以指数形式衰减，即发生了梯度消失（Gradient Vanishing）；
- 如果梯度大于1，那么随着层数增多，梯度更新将以指数形式膨胀，即发生梯度爆炸（Gradient Exploding）.

梯度消失和梯度爆炸主要存在RNN中，因为RNN中每个时间片使用相同的权值矩阵。对于一个DNN，虽然也涉及多个矩阵的相乘，但是通过精心设计权值的比例可以避免梯度消失和梯度爆炸的问题。


常用的用于解决梯度消失和梯度爆炸的方法如下所示：

- 处理梯度爆炸可以采用梯度截断的方法。

所谓梯度截断是指将梯度值超过阈值 \deta 的梯度手动降到 \deta 。虽然梯度截断会一定程度上改变梯度的方向，但梯度截断的方向依旧是朝向损失函数减小的方向。
对比梯度爆炸，梯度消失不能简单的通过类似梯度截断的阈值式方法来解决，因为长期依赖的现象也会产生很小的梯度。如果刻意提高小梯度的值将会使模型失去捕捉长期依赖的能力。

- 使用 ReLU、LReLU、ELU、maxout 等激活函数

sigmoid函数的梯度随着x的增大或减小和消失，而ReLU不会。

- Batch Normalization

通过规范化操作将输出信号x规范化到均值为0，方差为1保证网络的稳定性.Batch Normalization 就是通过对每一层的输出规范为均值和方差一致的方法，消除了参数w带来的放大缩小的影响，进而解决梯度消失和爆炸的问题。

Batch Normalization & Layer Normalization

<img src="https://github.com/ZhiweiZhang97/NLP/blob/main/image/Norm.png" width="400"/>

<img src="https://user-images.githubusercontent.com/Norm2.png" width="400"/>
