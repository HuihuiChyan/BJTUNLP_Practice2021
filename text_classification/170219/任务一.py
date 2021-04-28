import torch
from torch import nn, optim
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os
from collections import defaultdict
import re
import string
import matplotlib.pyplot as plt

EPOCH = 100     #训练轮次
MAX_LEN = 400  #训练集最大句子长度
BATCH_SIZE =128           # batch大小
EMBEDDING_SIZE = 300      # 词向量维度
LEARNING_RATE = 1e-4      # 学习率大小
KERNEL = (3, 4, 5)   # 卷积核长度
FILTER_NUM = 100          # 每种卷积核输出通道数
DROPOUT = 0.5             # dropout概率
TRAIN = False #是否是训练状态

#从文件读取数据
def readfile(filein, textlist):
    filenames = os.listdir(filein)
    for file in filenames:
        file = filein + file
        with open(file, "r", encoding = "utf-8") as f:
            word = f.read()
            f.close()
        textlist.append(str(word))

#读取测试集文件
def read_testfile(filein,textlist):
    with open(filein,'r',encoding = 'utf-8') as F:
        lines = F.readlines()
        for line in lines:
            line.strip()
            textlist.append(line)

#将句子分割成单词，预处理,并存为cleantext
def data_process(TextList,CleanPath):
    #WordsList = []
    with open(CleanPath,"w",encoding = "utf-8") as f:
        for i in range(len(TextList)):
            TextList[i] = TextList[i].lower()
            Words = re.split('[ ,.(){}!$\[\]<>"\'‘’~?#%@*`:/;_\-\n]',TextList[i])
            Words = DataClean(Words)
            #WordsList.append(Words)
            clean_Text = " ".join(Words)
            f.write(clean_Text)
            f.write("\n")
    #return WordsList

#读取干净数据，并转化为WordsList
def read_WordsList(FileName):
    WordsList = []
    with open(FileName, "r",encoding = "utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break;
            line = line.strip()
            Words = line.split()
            WordsList.append(Words)
        f.close()
    return WordsList

#数据预处理
def DataClean(WordList):
    punclist = [s for s in string.punctuation]
    new_WorldList = []
    for w in WordList:
        if (w not in punclist ) and  (w != ''):
            new_WorldList.append(w)
    return new_WorldList

#建立词表
def build_vocab(WordsList):
    words_index = {'UNK':0,'PAD':1}
    i = 2
    f=open('data/words_vocab.txt', 'w', encoding = 'utf-8')
    f.write('UNK 0\n')
    f.write('PAD 1\n')
    for Words in WordsList:
        for Word in Words:
            if (Word in words_index):
                continue
            words_index[Word] = i
            f.write(Word + ' ' + str(i) + '\n')
            i = i + 1
    f_glove = open('/home/zhangyu/分词/glove.model.6B.300d.txt','r',encoding = 'utf-8')
    Words =  [ o.split(" ")[0] for o in f_glove ]
    f_glove.close()
    for Word in Words:
        if (Word in words_index):
            continue
        words_index[Word] = i
        f.write(Word + ' ' + str(i) + '\n')
        i = i + 1
    f.close()
    return words_index

#读取词表文件
def read_vocab(FileName):
    embeddings_index = []
    with open(FileName,'r',encoding = 'utf-8') as f:
        def get_coefs(word, index): return word, int(index)
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in f)
    return embeddings_index

#用Glove预训练的模型建立Word embedding
def build_Wordembedding(word_index):
    EMBEDDING_FILE = "/home/zhangyu/分词/glove.model.6B.300d.txt"
    with open(EMBEDDING_FILE) as f:

        def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

        # 将所有的word作为key，numpy数组作为value放入字典
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    # 取出所有词向量，dict.values()返回的是dict_value, 要用np.stack将其转为ndarray
    all_embs = np.stack(embeddings_index.values())
    # 计算所有元素的均值和标准差
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    # 每个词向量的长度
    embed_size = all_embs.shape[1]

    nb_words = len(word_index)
    # 在高斯分布上采样， 利用计算出来的均值和标准差， 生成和预训练好的词向量相同分布的随机数组成的ndarray （假设预训练的词向量符合高斯分布）
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    #embedding_vocab = {}

    # 利用预训练的词向量的均值和标准差为数据集中的词随机初始化词向量
    # 然后再使用预训练词向量中的词去替换随机初始化数据集的词向量。
    #f=open('data/words_vocab.txt', 'w', encoding = 'utf-8')
    for word, i in word_index.items():
        if (word == 'PAD'):
            embedding_matrix[i] = [0]*len(embedding_matrix[i])
        # 如果dict的key中包括word， 就返回其value。 否则返回none。
        embedding_vector = embeddings_index.get(word)
        # 词在数据集中和训练词向量的数据集中都出现，使用预训练的词向量替换随机初始化的词向量
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        #embedding_vocab[word] = embedding_matrix[i]
        #f.write(word + ' ' + " ".join('%s' %value for value in embedding_vocab[word]) + '\n')
    return embedding_matrix

#生成输入数据
def process_input(WordsList, words_vocab,truncate=False):
    vocab = defaultdict(lambda:0)
    for word in words_vocab:
        if word == 'UNK':
            continue
        vocab[word]=words_vocab[word]

    new_WordsList = [[vocab[word] for word in Words] for Words in WordsList]
    #if truncate:  # 只针对训练集，不可用于测试集
        #new_WordsList = [line[:MAX_LEN] for line in new_WordsList]

    new_WordsList = [line[:MAX_LEN] for line in new_WordsList]
    len_lines = [len(line) for line in new_WordsList]
    max_len = max(len_lines)
    #print('max_len='+str(max_len))
    new_WordsList = [line + [vocab['PAD'] for i in range(max_len - len(line))] for line in new_WordsList]
    #new_WordsList = np.array(new_WordsList)
    #new_WordsList = rnn_utils.pad_sequence(new_WordsList, batch_first=True, padding_value=1)
    return new_WordsList,len_lines

#TextCNN模型
class TextCNN(nn.Module):
    # output_size为输出类别（2个类别，0和1）,三种kernel，size分别是3,4，5，每种kernel有100个
    def __init__(self, vocab_size, embedding_dim, output_size, filter_num = 100, kernel_list = KERNEL_LIST, dropout = DROPOUT):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 1表示channel_num，filter_num即输出数据通道数，卷积核大小为(kernel, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, filter_num, (kernel, embedding_dim)),
                          nn.LeakyReLU(),
                          nn.MaxPool2d((TRAIN_MAX_SENTENCE_LEN - kernel + 1, 1)))
            for kernel in kernel_list
        ])
        self.fc = nn.Linear(filter_num * len(kernel_list), output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        x = x.unsqueeze(1)  # (batch, channel_num, seq_len, embedding_dim)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)  # [64, 500, 1, 1]，各通道的数据拼接在一起
        out = out.view(x.size(0), -1)  # 展平
        out = self.dropout(out)  # 构建dropout层
        logits = self.fc(out)  # 结果输出[64, 2]
        return logits

#准确率每30个epoch，下降至原来的10%
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LEARNING_RATE * (0.5 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

#加载数据
def create_dataloader(X, Y = [0], shuffle=True):
    '''tmp = []
    for x in X:
        tmp.append(torch.tensor(x).long())
    X = tmp     # python的List转化为了pytorch的tensor
    X = rnn_utils.pad_sequence(X, batch_first=True, padding_value=1)'''
    X = torch.LongTensor(X)
    #Y = torch.tensor(Y)
    if (Y != [0]):
        Y = torch.tensor(Y)
        dataset = data.TensorDataset(X, Y)
    else :
        dataset = X
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=shuffle)
    return dataloader

#获取batch数据
def get_batch(x, y, batch_size=BATCH_SIZE, shuffle=True):
    x = np.array(x)
    y = np.array(y)
    assert x.shape[0] == y.shape[0], "error shape!"
    if shuffle:   #排序
        shuffled_index = np.random.permutation(range(x.shape[0]))
        x = x[shuffled_index]
        y = y[shuffled_index]

    n_batches = int(x.shape[0] / batch_size)  # 统计共几个完整的batch
    for i in range(n_batches - 1):
        x_batch = x[i*batch_size: (i + 1)*batch_size]
        y_batch = y[i*batch_size: (i + 1)*batch_size]
        yield x_batch, y_batch
#计算准确率
def binary_acc(preds, y):
    """
    计算二元结果的准确率
    :param preds:比较值1
    :param y: 比较值2
    :return: 相同值比率
    """
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc

#计算acc 值
def cal_acc(predict_Y, real_Y):
    #real_predict_Y = []  # 真实的输出
    #real_real_Y = []  # 真实的标签
    #for line in zip(predict_Y, real_Y, lens):
     #   real_predict_Y.extend(line[0][:line[2]])
     #   real_real_Y.extend(line[1][:line[2]])
    acc = accuracy_score(real_Y, predict_Y)
    #f1_value = f1_score(real_Y, predict_Y, average='macro')
    return acc

#读取文件，数据预处理
pos_train = []
neg_train = []
pos_dev = []
neg_dev = []
readfile("G:/研究生入学前任务/任务一/aclImdb/train/neg/",neg_train)
readfile("G:/研究生入学前任务/任务一/aclImdb/train/pos/",pos_train)
readfile("G:/研究生入学前任务/任务一/aclImdb/dev/neg/",neg_dev)
readfile("G:/研究生入学前任务/任务一/aclImdb/dev/pos/",pos_dev)
#neg_train_WordsList = TextList_To_WordsList(neg_train)
#pos_train_WordsList = TextList_To_WordsList(pos_train)
data_process(neg_train,'data/train/neg_clean.txt')
data_process(pos_train,'data/train/pos_clean.txt')
data_process(neg_dev,'data/dev/neg_clean.txt')
data_process(pos_dev,'data/dev/pos_clean.txt')

#读取测试集数据&预处理
#testList = []
#read_testfile('data/test.txt',testList)
#data_process(testList,'data/test_clean.txt')

#读取预处理后的文件
neg_train_WordsList = read_WordsList('data/train/neg_clean.txt')
pos_train_WordsList = read_WordsList('data/train/pos_clean.txt')
neg_dev_WordsList = read_WordsList('data/dev/neg_clean.txt')
pos_dev_WordsList = read_WordsList('data/dev/pos_clean.txt')
test_WordList = read_WordsList('data/test_clean.txt')

#生成词表
'''WordsList = neg_train_WordsList[:]
WordsList.extend(pos_train_WordsList)
words_index = build_vocab(WordsList)
print(len(words_index))'''
#build_vocab(words_index)

words_vocab = read_vocab('data/words_vocab.txt')
#生成word embedding
word_embedding=build_Wordembedding(words_vocab)
#print(word_embedding[1])
#print(word_embedding.shape)

#print(words_vocab)
train_WordsList = neg_train_WordsList[:]
train_WordsList.extend(pos_train_WordsList)
dev_WordsList = neg_dev_WordsList[:]
dev_WordsList.extend(pos_dev_WordsList)

X_train,train_lens = process_input(train_WordsList,words_vocab,True)
Y_train = [0]*len(neg_train_WordsList)
Y_train.extend([1]*len(pos_train_WordsList))
X_dev,dev_lens = process_input(dev_WordsList,words_vocab)
Y_dev = [0]*len(neg_dev_WordsList)
Y_dev.extend([1]*len(pos_dev_WordsList))
X_test,_ = process_input(test_WordList,words_vocab)
x_test_arr = np.array(X_test)
#print(x_test_arr.max())
#print(x_test_arr.min())

#构建模型
cnn_Model = TextCNN(len(words_vocab), EMBEDDING_SIZE, 2).cuda()
cnn_Model.embedding.weight.data.copy_(torch.FloatTensor(word_embedding))
# 优化器和交叉损失函数
#optimizer = optim.Adam(cnn_Model.parameters(), lr=LEARNING_RATE)
optimizer = optim.Adam(cnn_Model.parameters(), lr=LEARNING_RATE, weight_decay = 0.0005)
loss_function = nn.CrossEntropyLoss()

#训练 & 验证
if __name__ == "__main__":
    if TRAIN:
        global_step = 0  # 进行一次梯度反传，叫做一个step
        best_acc = 0.0
        best_step = 0
        train_dataloader = create_dataloader(X_train,Y_train)
        dev_dataloader = create_dataloader(X_dev, Y_dev)
        loss_list=[]
        cnn_dev_acc=[]
        cnn_train_acc=[]
        for epoch in range(EPOCH):
            #每25个epoch重新划分训练集和验证集，防止过拟合
            # 模型训练
            #train_acc = train(cnn, optimizer, criterion)
            avg_train_acc=[]
            cnn_Model.train()  # 开启训练模式
            if (epoch % 50  == 0):
                lr = adjust_learning_rate(optimizer, epoch)
                print('epoch = %d, lr = %.6f',(epoch, lr))
            for batched_X, batched_Y in train_dataloader:
                batched_output = cnn_Model(batched_X.cuda())  # 执行了我们刚刚定义的forward()函数
                #print(batched_output.grad)
                # 计算损失值
                loss = loss_function(batched_output.cpu(), batched_Y)
                loss_list.append(loss)
                optimizer.zero_grad()  # 清零累积的梯度
                loss.backward()  # 梯度反向传播
                optimizer.step()  # 梯度更新

                batched_output = torch.argmax(batched_output, dim=1)
                batched_pre = batched_output.tolist()
                batched_real = batched_Y.tolist()
                train_acc = cal_acc(batched_pre, batched_real)
                avg_train_acc.append(train_acc)

                global_step += 1
                if global_step % 100 == 0:
                    print('train step %d, loss is %.4f,train acc = %.4f' % (global_step, loss,train_acc))
                    #print(batched_output.grad)
                if global_step % 500 == 0:
                    # evaluation!
                    cnn_Model.eval()
                    predict_Y = []  # 所有的预测类别
                    real_Y = []  # 所有的真实类别
                    lens = []  # 所有的句子长度
                    for batched_X, batched_Y in dev_dataloader:
                        batched_output = cnn_Model(batched_X.cuda())
                        batched_logits = torch.argmax(batched_output.cpu(), dim=1)

                        predict_Y.extend(batched_logits.tolist())  # pytorch的tensor转化成python的list
                        real_Y.extend(batched_Y.tolist())
                        #lens.extend(batched_lens.tolist())

                    # 计算acc和F1值
                    #f1_value, acc = cal_f1_acc(predict_Y, real_Y, lens)
                    acc = cal_acc(predict_Y, real_Y)
                    cnn_dev_acc.append(acc)
                    #cnn_dev_f1.append(f1_value)
                    print('eval step %d, accuracy is %.4f' % (global_step, acc))

                    if acc > best_acc:
                        best_acc = acc
                        best_step = global_step
                        torch.save(cnn_Model, 'best_model_TextCNN.bin')
            avg_train_acc = np.array(avg_train_acc).mean()
            cnn_train_acc.append(avg_train_acc)

        print('All training finished. Best step is %d, best acc is %.4f' % (best_step, best_acc))
    else:   #测试
        test_dataloader = create_dataloader(X_test,[0],False)
        print("Test start...")
        cnn_Model = torch.load('best_model_TextCNN.bin')
        cnn_Model.eval()
        predict_Y = []  # 所有的输出
        X_test_result = []
        real_Y = []  # 所有的标签
        lens = []  # 所有的句子长度
        for batched_X in test_dataloader:
            #batched_X = torch.LongTensor(batched_X)
            batched_output = cnn_Model(batched_X.cuda())
            batched_logits = torch.argmax(batched_output.cpu(), dim=-1)
            X_test_result.extend(batched_X.tolist())
            predict_Y.extend(batched_logits.tolist())  # pytorch的tensor转化成python的list
        with open('data/test_result.txt', 'w', encoding='utf-8') as f:
            for x in predict_Y:
                f.write(str(x) + '\n')
        f.close()