### 概况

模型主要实现了模型ESIM的结构，包括论文所提到的input encoding、local inference modeling和inference composition部分，文本匹配最终测试准确率为86.41%。

### 词向量

使用glove的300维的预训练模型进行初始化，其中PAD设置为全零，UNK根据词向量的高斯分布随机初始化。

### 数据预处理

采用标点和空格对文本进行分词，将大写字母全部转换为小写。
label进行ID化时，将非neutral、contradiction、entailment的词的ID设为0（ID为0对应为neutral）。

### 遇到的问题

#### 震荡和不收敛
由于一开始设置的batch_size太小，导致训练准确率出现严重震荡，loss下降缓慢且不收敛，后将batch_size调大，并调节学习率等其他参数后，模型效果逐渐变好。

### 模型改进

将最后的全连接层由一层全连接层改为三层，并使用了Relu激活函数和dropout，在自带测试集上的准确率由85.28%提升至85.63%

### 模型参数

```python
EPOCH = 20     #训练轮次
MAX_LEN = 20   #句子最大长度
BATCH_SIZE = 512          # batch大小
HIDDEN_SIZE = 300         # 隐藏层大小
LEARNING_RATE = 2e-4      # 学习率大小
EMBEDDING_SIZE = 300      # 词向量维度
DROPOUT = 0.5             # dropout概率
CLASS_NUM = 3             # 类别数量
```
### 改进想法

1.修改attention机制中mask种类，观察是否会有提升；  
3.调整weight_decay参数的大小，避免过拟合，提升验证集准确率。  
