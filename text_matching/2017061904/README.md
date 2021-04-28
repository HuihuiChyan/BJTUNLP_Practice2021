#### 概况

实现了基本的ESIM模型，在此基础上调整了最后一层多层感知机的结构，使得模型的准确率有所提高。还在拼接层后加了三层线性层，进一步使模型能更好的进行预测

测试集准确率为85.30%

#### 数据处理

并没有处理 “-” 的数据，将其作为其中一种结果进行训练；

对一些符号进行分割处理，例如括号和引号之类

#### 遇到的问题

主要问题是搭建完基本的ESIM模型后，训练的准确率一直维持在75%左右。

通过逐步修改完善最后一层多层感知机以及添加三层线性层逐渐将准确率提高到85%左右。

#### 模型参数

```python
batch_size = 128
emvbedding_dim = 300
hidden_dim = 200
learning_rate = 0.0001
epoch = 50 
```

#### 后期展望

1. 通过mask机制消除padding部分的影响
2. 处理 ”-“ 的数据，进一步观察模型训练结果