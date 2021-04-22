#### 说明：

1. 全体新加入研究组的同学，需要认真完成自己的毕设以及三个练习任务，完成的情况会反馈给导师，作为进一步培养的依据；
2. 在模型写完并完成训练后，请联系程序练习负责人获取测试集，然后用自己训好的模型推断获得结果，处理成要求的格式，命名为result.txt，发给负责人，负责人会计算测试结果；
3. 最后将全部代码+README.md说明文件打包成以自己学号命名的压缩包发给负责人；
4. 可以参考开源代码，但是最后提交的代码必须是自己写的；
5. 禁止使用除了训练集以外的任何数据用于训练，禁止使用验证集、测试集进行训练。

### 大家的毕业设计版本迭代记录

1. 请每个任务涉及到的同学使用同样的测试集（训练集可以根据自己的毕设题目做调整），并且禁止将测试集也用于训练；
2. 每一次大家对模型、训练策略进行改进后，都需要在测试集上测试，然后反馈给我，我会记录大家版本迭代的情况；

## 程序练习提交情况（不断更新中）

序号按照首次提交的时间排列。

#### 文本分类：

| 序号 | 学号       | 准确率 | 最终提交时间      | 说明  |
| ---- | ----------| ------ | -------------|------|
| 1    | 17231190  | 90.63% | 2021年01月10日||
| 2    | 2017061904  | 89.17% | 2021年01月13日||
| 3    | 20171002495  | 89.28% | 2021年01月20日||
| 4    | 1707020318  | 90.31% | 2021年01月21日||
| 5    | 17271137 | 89.98% |2021年02月09日  |
| 6    | 170219 | 89.28% |2021年03月28日  | 

#### 文本匹配：

| 序号 | 学号       | 准确率 | 最终提交时间      | 说明  |
| ---- | ----------| ------ | -------------|------|
| 1    | 17231190  | 85.54% | 2021年01月10日||
| 2    | 20171002495  | 86.41% | 2021年03月01日||
| 3    | 2017061904  | 85.30% | 2021年03月18日||


### 任务一：基于TextCNN的文本分类

数据集：[Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) 

参考论文：Convolutional Neural Networks for Sentence Classification，https://arxiv.org/abs/1408.5882

需要了解的知识点：

1. 文本特征表示：词向量
   1. 对word embedding随机初始化
   2. 用glove预训练的embedding进行初始化 https://nlp.stanford.edu/projects/glove/
2. CNN如何提取文本的特征

模型图：

<img src="TextCNN.PNG" width="600"> 

说明：

1. 训练集25000句，测试集25000句，需要自己写脚本合在一起；
2. 请将训练集用于训练，测试集用于验证，最后我会再给你一个测试集；
2. 测试结果格式：每行对应一句话的分类结果；

当前的SOTA排名：https://github.com/sebastianruder/NLP-progress/blob/master/english/sentiment_analysis.md

### 任务二：基于ESIM的文本匹配

输入两个句子，判断它们之间的关系。参考ESIM（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现。

数据集：https://nlp.stanford.edu/projects/snli/

参考论文：Enhanced LSTM for Natural Language Inference，<https://arxiv.org/pdf/1609.06038v3.pdf>

模型图：

<img src="ESIM.png" width="600"> 

知识点：

1. 注意力机制在NLP中的应用

说明：

1. 训练集、验证集、测试集已经分割好了，但是你仅使用训练集和验证集即可，最后我会再给你一个测试集；
2. 测试结果格式：每行对应一个句对的匹配结果；

当前的SOTA排名：https://nlp.stanford.edu/projects/snli/

### 任务三：基于Bert的自然语言理解

Bert可以用来进行分类、标注、匹配等多种自然语言理解任务。这里需要用Bert重新实现上述两个任务中的任意一个。

建议使用的框架：Huggingface，https://github.com/huggingface/transformers

参考论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding，https://arxiv.org/abs/1810.04805

模型图：

<img src="BERT.jpg" width="550"> 

知识点：

1. 预训练和预训练模型
2. 子词切分
3. 自注意力和transformer（不过不需要你自己写模型）
