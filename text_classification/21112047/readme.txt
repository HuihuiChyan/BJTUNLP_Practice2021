概况：
模型主要实现了TextCNN模型，测试集准确率为0.8985。

词向量：
embedding采用glove6B的维度为300的预训练模型进行词向量的初始化。

数据预处理：
首先将所有句子文件合并，然后对数据进行清洗。

参数设置：
batch_size = 30
learn_rate = 2e-4
max_len = 400
embedding_size = 300
kernel_size = [3, 4, 5]
out_channels = 300
dropout_prob = 0.5
epochs_num =15