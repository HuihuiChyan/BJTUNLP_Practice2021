1.数据预处理
（1）设置sentence_max_len = 500，对句子进行截断。不足的用‘pad’补齐，未知词用‘unk’替代，均适用glove中对应词向量。
（2）对isn't don't 等缩写进行展开替换。
（3）去除标点符号。
（4）使用jieba库对句子分词生成列表。
（5）尝试对高频词汇the 等进行停用，效果不明显。

2.参数设置
batch_size=64
filter_num=512
epoch=50
kernel_list=[2,3,4,5]
label_size=2
lr=1e-4
weight_decay=1e-3
Dropout(0.5)

3.遇到的问题
在使用dropout时，起初直接使用了F.dropout(0.5)。通过检查及csdn查阅发现F.dropout()的参数training默认为false，导致并没起作用。因此accuracy一直没有太大提升。所以最开始对很多参数的做出的尝试并不准确。