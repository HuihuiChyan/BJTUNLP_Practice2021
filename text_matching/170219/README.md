**概况**

模型实现了基本的ESIM的结构，在此基础上使用了三层全连接层，使得模型的准确率有所提高，测试集准确率为86.35% 

**数据预处理**

Label为“-”时，将ID设为0（ID为0对应为neutral）。

对一些符号进行分割处理（所有可能遇到的基本的标点符号）。

使用GloVe的300维的预训练好的词向量模型，PAD设置为全0，UNK随机初始化给。

**遇到的问题**

将最后的全连接层由一层全连接层改为三层，并使用了Relu激活函数和dropout，准确率从85%提到了86%

**模型参数**

EPOCH = 50   #训练轮次

BATCH_SIZE = 256     # batch大小

HIDDEN_SIZE = 300     # 隐藏层大小

LEARNING_RATE = 1e-4   # 学习率大小

EMBEDDING_SIZE = 300   # 词向量维度

DROPOUT = 0.5       # dropout概率

CLASS_NUM = 3       # 类别数量

**改进想法**

修改attention机制中mask种类，观察是否会有提升；
