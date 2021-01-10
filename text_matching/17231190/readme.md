### 词向量：
使用300d的glove词向量，unk，pad均使用其中自带的。

### 数据处理：
lower 和 Tokenizer把符号分开。

label处理：- 认为作为无关（neutral）的处理比较合理，

### 模型：
tree-LSTM 感觉写不出来，直接放弃了。

在模型搭建时，linear，bilstm等中都设置了dropout。

在学习注意力机制的时候，有些代码使用了mask机制，就是句子在填充到一定长度时，两个句子相乘后，填充部分经过softmax后占有了概率，mask机制考虑消除这一影响。

使用mask机制，
test_accuracy 0.846100    dev_accuracy 0.855100

未使用mask机制：
test_accuracy 0.846700    dev_accuracy 0.859300

### 可能的改进：
1.学习率和优化器参数。

2.mask没起到作用可能是padding的时候不是进行的全零填充。给一个大的负的偏执项还会影响模型的效果
