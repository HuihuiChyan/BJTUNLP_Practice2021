概况：
模型主要实现了ESIM模型，测试集准确率为0.8010。

数据预处理：
从数据中，提取并分离出text1、text2、label，label中存在“－”，将这种数据舍弃。

词向量：
embedding采用glove840B的维度为300的与训练模型进行词向量的初始化。

参数设置：
batch_size = 200
learn_rate = 6e-4
max_len = 96
embedding_size = 300
hidden_size = 300
class_num = 3
epochs_num =50