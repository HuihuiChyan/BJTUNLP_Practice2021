### 概况

主要实现了 TextCNN 的模型，测试集准确率达到了 90.31%。

### 词向量

使用 glove 的 300 维的预训练模型进行初始化，其中 ``pad`` 与 ``unk`` 均使用 glove 自带向量，idx 分别为 ``10109`` 、``201534``。

###  数据预处理

读入数据后，单词全部变为小写，对 ``'is、're`` 等缩写进行了还原，采用正则对数据预处理，去除 `` []{}()``，并对 ``.,!?"`` 五个标点进行前后分割。

### 遇到的问题和失误

#### 未对 ``pad`` 和 ``unk`` 进行处理

起初预处理时，词典仅单纯的调用 torchtext 包中的 vocab 工具生成，未在词典中的词没有处理，``pad`` 设置为 0，导致对于不同的正则表达式，验证集准确率浮动较大。

### 做出的改进

输入在 embedding 层与卷积层中间，加入了 BiLSTM 层，BiLSTM 的输入为其中论文中非固定的 embedding，并将 BiLSTM 输出结果与论文中的两个 embedding 进行拼接，形成三层 channel，再进行后续计算。

测试准确率从 $89.55\%$ 提升至 $90.31\%$ 。

### 可能的改进

1. 将池化层取最大值换为取前 K 大值。
2. 不知道去掉一些停用词对准确率的提升是否有帮助。

### 模型参数

```python
batch_size = 64
max_len = 500
embed_size = 300
kernel_sizes = [2, 3, 4, 5]
num_channels = [100, 100, 100, 100]
dropout_p = 0.5
epoch_num = 30
lr = 1e-4
weight_decay = 1e-7
```

