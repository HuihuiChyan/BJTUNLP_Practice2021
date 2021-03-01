import re
import string
import torch
from torch.utils import data
import numpy as np
from ESIM import *
from collections import defaultdict
from sklearn.metrics import accuracy_score

#宏定义
EPOCH = 20     #训练轮次
MAX_LEN = 20   #句子最大长度
BATCH_SIZE = 512          # batch大小
HIDDEN_SIZE = 300         # 隐藏层大小
LEARNING_RATE = 2e-4      # 学习率大小
EMBEDDING_SIZE = 300      # 词向量维度
DROPOUT = 0.5             # dropout概率
CLASS_NUM = 3             # 类别数量
TRAIN_STATE = True        # 是否是训练状态

#config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#初次数据处理
def dataclean(filename,fileout):
    fout = open(fileout, 'w', encoding = 'utf-8')
    with open(filename, 'r', encoding = 'utf-8') as fin:
        fin.readline()
        lines = fin.readlines()
        for line in lines:
            split_line = line.split('\t')
            sentence1 = split_line[5]
            fout.write(sentence1 + '\t')
            sentence2 = split_line[6]
            fout.write(sentence2 + '\t')
            gold_label = split_line[0]
            fout.write(gold_label + '\n')
    fin.close()
    fout.close()

#读取处理后的文件，分词
def readfile(filename, test=False):
    sentence1_list = []
    sentence2_list = []
    label_list = []
    with open(filename,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.lower()
            split_line = line.split('\t')
            words = re.split('[ ,.(){}!$\[\]<>"\'‘’~?#%@*`:/;_\-\n]', split_line[0])
            Words = []
            for word in words:
                if ( word != ''):
                    Words.append(word)
            #Words = [word for word in words and word !='']
            sentence1_list.append(Words)
            words = re.split('[ ,.(){}!$\[\]<>"\'‘’~?#%@*`:/;_\-\n]', split_line[1])
            Words = []
            for word in words:
                if (word != ''):
                    Words.append(word)
            sentence2_list.append(Words)
            if test != True:
                label_list.append(split_line[2].strip())
    f.close()

#建立词表
def build_vocab(traindata):
    vocabSet = set([])
    for data in traindata:
        vocabSet = vocabSet | set(data)  # 取并集
    f_glove = open('data/glove.6B.300d.txt', 'r', encoding='utf-8')
    Words = [o.split(" ")[0] for o in f_glove]
    f_glove.close()
    vocabSet = vocabSet | set(Words)
    vocab = list(vocabSet)

    i = 2   #从2开始
    f = open('data/Vocab.txt','w',encoding='utf-8')
    for word in vocab:
        f.write(word + ' ' + str(i) + '\n')
        i = i + 1
    f.close()
    return sentence1_list, sentence2_list, label_list

#读取词表
def read_vocab(filename):
    #vocab = defaultdict(lambda:0)
    vocab = {'UNK':0,'PAD':1}
    #vocab['UNK'] = 0
    #vocab['PAD'] = 1
    with open(filename,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            vocab[line[0]] = line[1]
    f.close()
    return vocab

#生成emdedding
def build_Wordembedding(Vocab):
    word_index =Vocab.copy()
    #lon = len(word_index)
    #word_index['UNK'] = 0
    EMBEDDING_FILE = "./data/glove.6B.300d.txt"
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
            embedding_matrix[int(i)] = embedding_vector
    return embedding_matrix
#生成输入数据
def create_iput(sen1, sen2, labels, vocab, labelid, train = False):
    result_sen1 = [[int(vocab[word]) if word in vocab else vocab['UNK'] for word in words] for words in sen1]
    result_sen2 = [[int(vocab[word]) if word in vocab else vocab['UNK'] for word in words] for words in sen2]
    result_labels = [int(labelid[label]) for label in labels]
    if train:
        result_sen1 = [line[:MAX_LEN] for line in result_sen1]
        result_sen2 = [line[:MAX_LEN] for line in result_sen2]
    lens1 = [len(s) for s in result_sen1]
    lens2 = [len(s) for s in result_sen2]
    lens1.extend(lens2)
    max_len = max(lens1)
    result_sen1 = [line + [vocab['PAD'] for i in range(max_len - len(line))] for line in result_sen1]
    result_sen2 = [line + [vocab['PAD'] for i in range(max_len - len(line))] for line in result_sen2]
    return result_sen1,result_sen2,result_labels

#生成batch数据
def create_dataloader(sen1,sen2,label = None, shuffle = True):
    sen1 = torch.LongTensor(sen1)
    sen2 = torch.LongTensor(sen2)
    if (label != None):
        label = torch.tensor(label)
        dataset = data.TensorDataset(sen1, sen2, label)
    else:
        dataset = data.TensorDataset(sen1, sen2)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=shuffle)
    return dataloader



if __name__ == "__main__":

    #预处理数据
    #dataclean('data/snli_1.0_train.txt','data/traindata.txt')
    #dataclean('data/snli_1.0_test.txt','data/testdata.txt')
    #dataclean('data/snli_1.0_dev.txt','data/devdata.txt')

    #读取文件
    train_s1, train_s2,train_labels = readfile('data/traindata.txt')
    dev_s1, dev_s2, dev_labels = readfile('data/devdata.txt')
    test_s1, test_s2, test_labels = readfile('data/testdata.txt',True)

    #将测试标签写入文件
    '''with open('data/testreal.txt', 'w', encoding = 'utf-8') as f:
        for label in test_labels:
            f.write(label + '\n')
    f.close()'''

    #建立词表
    '''traindata = []
    traindata = train_s1[:]
    traindata.extend(train_s2)
    build_vocab(traindata)'''

    #读取词表
    Vocab = read_vocab('data/Vocab.txt')
    vocab_size = len(Vocab)

    #生成embedding
    word_embedding = build_Wordembedding(Vocab)
    vocab_size = len(Vocab)

    #生成输入数据
    label_id = defaultdict(lambda:0)
    label_id['neutral'] = 0
    label_id['contradiction'] = 1
    label_id['entailment'] =2
    train_sen1,train_sen2,train_labels = create_iput(train_s1,train_s2,train_labels,Vocab,label_id)
    vocab_size = len(Vocab)
    dev_sen1,dev_sen2,dev_labels = create_iput(dev_s1,dev_s2,dev_labels,Vocab,label_id)
    vocab_size = len(Vocab)
    test_sen1,test_sen2,test_labels = create_iput(test_s1,test_s2,test_labels,Vocab,label_id)


    vocab_size = len(Vocab)

    #训练&测试
    esim_model = ESIM(vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, CLASS_NUM, DROPOUT,word_embedding).cuda()
    optimizer = optim.Adam(esim_model.parameters(), lr=LEARNING_RATE, weight_decay = 1e-6) #, weight_decay = 1e-7
    loss_function = nn.CrossEntropyLoss(reduce=True,size_average=True)
    if TRAIN_STATE:
        global_step = 0
        best_acc = 0.0
        best_step = 0
        train_dataloader = create_dataloader(train_sen1, train_sen2,train_labels,True)
        dev_dataloader = create_dataloader(dev_sen1, dev_sen2,dev_labels)
        fd = open('data/dev_acc.txt', 'w', encoding = 'utf-8')
        ft = open('data/train_loss_acc.txt', 'w', encoding = 'utf-8')
        loss_list = []
        cnn_dev_acc = []
        cnn_train_acc = []
        for epoch in range(EPOCH):
            # 模型训练
            for batch_sen1,batch_sen2,batch_label in train_dataloader:
                esim_model.train()  # 开启训练模式
                batch_out = esim_model(batch_sen1.cuda(), batch_sen2.cuda())
                loss = loss_function(batch_out.cpu(),batch_label.squeeze(-1))  #计算损失值
                optimizer.zero_grad() #清零累积的梯度
                loss.backward()       #梯度反向传播
                optimizer.step()      #梯度更新

                batch_out = torch.argmax(batch_out, dim=1)
                batch_pre = batch_out.tolist()
                batch_real = batch_label.tolist()
                train_acc = accuracy_score(batch_pre, batch_real)
                #avg_train_acc.append(train_acc)
                global_step += 1
                if global_step % 50 == 0:
                    print('train step %d, loss is %.4f,train acc = %.4f' % (global_step, loss, train_acc))
                    ft.write(str(float(loss)) +' ' + str(train_acc) + '\n')
                if global_step % 500 == 0:
                    # 模型验证
                    esim_model.eval()
                    pre_Y = []
                    real_Y = []
                    #loss_list = []
                    for batch_devsen1,batch_devsen2,batch_devlabel in dev_dataloader:
                        batch_devout = esim_model(batch_devsen1.cuda(), batch_devsen2.cuda())
                        batch_devlogits = torch.argmax(batch_devout.cpu(), dim = 1)
                        pre_Y.extend(batch_devlogits.tolist())
                        real_Y.extend(batch_devlabel.tolist())
                    acc = accuracy_score(pre_Y, real_Y)
                    print('eval step %d, accuracy is %.4f' % (global_step, acc))
                    fd.write(str(acc) + '\n')
                    if acc > best_acc:
                        best_acc = acc
                        best_step = global_step
                        torch.save(esim_model,'best_Mondel.bin')
        print('All training finished. Best step is %d, best acc is %.4f' % (best_step, best_acc))
        fd.close()
        ft.close()
    else:
        # 模型测试
        print("Test start...")
        esim_model = torch.load('best_Mondel.bin')
        esim_model.eval()
        test_dataloader = create_dataloader(test_sen1, test_sen2, shuffle = False)
        pre_Y = []
        for batch_sen1,batch_sen2 in test_dataloader:
            batch_out = esim_model(batch_sen1.cuda(), batch_sen2.cuda())
            batch_logits = torch.argmax(batch_out.cpu(), dim = 1)
            pre_Y.extend(batch_logits.tolist())
        num_to_label = {0:'neutral',1:'contradiction',2:'entailment'}
        test_pre = [num_to_label[i] for i in pre_Y]
        with open('data/testpre.txt', 'w', encoding = 'utf-8') as f:
            for label in test_pre:
                f.write(label + '\n')
        print("Test end...")
