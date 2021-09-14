参数：
seq_len = 128
batch_size = 256
embedding_size = 300
hidden_size = 300
dropout = 0.5
lr = 1e-3

词向量：使用了glove训好的6B 300D向量。
跑到第六个epoch时，验证集结果到了87.09，使用原本的test集测试准确率为86.45.

遇到的问题：lr调参过大导致模型不断震荡，预测正确率在1/3不断徘徊；最开始使用word_embedding时忘记引入glove向量而是直接初始化，则效果较差。