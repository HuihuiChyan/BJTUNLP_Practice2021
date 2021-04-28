import re
import pdb
from torch.utils import data
import numpy as np
from ESIM_model import *
from collections import defaultdict
from sklearn.metrics import accuracy_score

#宏定义
EPOCH = 50     #训练轮次
MAX_LEN = 20   #句子最大长度
BATCH_SIZE = 256          # batch大小
HIDDEN_SIZE = 300         # 隐藏层大小
LEARNING_RATE = 1e-4      # 学习率大小
EMBEDDING_SIZE = 300      # 词向量维度
DROPOUT = 0.5             # dropout概率
CLASS_NUM = 3             # 类别数量
TRAIN = True        # 是否是训练状态

#config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#初次数据处理
def data_process(filename,fileout):
    with open(filename, 'r', encoding = 'utf-8') as fin,\
    open(fileout, 'w', encoding = 'utf-8') as fout:
        lines = fin.readlines()
        for line in lines:
            realline = line.split('\t')
            premise = realline[5]
            fout.write(premise + '\t')
            hypothesis = realline[6]
            fout.write(hypothesis + '\t')
            label = realline[0]
            fout.write(label + '\n')
    fin.close()
    fout.close()

#读取处理后的文件，分词
def create_list(filename, test=False):
    premise_list = []
    hypothesis_list = []
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
            premise_list.append(Words)
            words = re.split('[ ,.(){}!$\[\]<>"\'‘’~?#%@*`:/;_\-\n]', split_line[1])
            Words = []
            for word in words:
                if (word != ''):
                    Words.append(word)
            hypothesis_list.append(Words)
            if test != True:
                label_list.append(split_line[2].strip())
    f.close()
    return premise_list, hypothesis_list, label_list

#建立词表
def build_vocab(traindata):
    vocabset = set([])
    for data in traindata:
        vocabset = vocabset | set(data)  # 取并集
    with open('/home/zhangyu/分词/glove.model.6B.300d.txt', 'r', encoding='utf-8') as f_glove:
        glove=f_glove.readlines()
        Words = [o.split(" ")[0] for o in glove[1:]]
        f_glove.close()
    vocabSet = vocabset | set(Words)
    vocab = list(vocabSet)

    i = 2   #从2开始
    f = open('data/Vocab.txt','w',encoding='utf-8')
    for word in vocab:
        f.write(word + ' ' + str(i) + '\n')
        i = i + 1
    f.close()

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
    glove_file= "/home/zhangyu/分词/glove.model.6B.300d.txt"
    with open(glove_file, 'r',encoding='utf-8') as f:
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')
        # 将所有的word作为key，numpy数组作为value放入字典
        glove=f.readlines()[1:]
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in glove)

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
    for i, word in enumerate(word_index):

        if (word == 'PAD'):

            embedding_matrix[i] = [0]*len(embedding_matrix[i])
        # 如果dict的key中包括word， 就返回其value。 否则返回none。
        embedding_vector = embeddings_index.get(word)
        # 词在数据集中和训练词向量的数据集中都出现，使用预训练的词向量替换随机初始化的词向量
        if embedding_vector is not None:
            embedding_matrix[int(i)] = embedding_vector

    return embedding_matrix

#生成输入数据
def create_input(sen1, sen2, labels, vocab, label_id, train = False):
    result_sen1 = [[int(vocab[word]) if word in vocab else vocab['UNK'] for word in words] for words in sen1]
    result_sen2 = [[int(vocab[word]) if word in vocab else vocab['UNK'] for word in words] for words in sen2]
    result_labels = [int(label_id[label]) for label in labels]
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    host = torch.device('cpu')
    #预处理数据
    #dataclean('snli_1.0/snli_1.0_train.txt','data/traindata.txt')
    #dataclean('snli_1.0/snli_1.0_test.txt','data/testdata.txt')
    #dataclean('snli_1.0/snli_1.0_dev.txt','data/devdata.txt')

    #读取文件
    print(20*'=','加载数据......',20*'=')
    train_p, train_h,train_labels = create_list('data/traindata.txt')
    dev_p, dev_h, dev_labels = create_list('data/devdata.txt')
    test_p, test_h, test_labels = create_list('data/testdata.txt',True)
    print(20*'=','加载完成......',20*'=')
    #将测试标签写入文件
    '''with open('data/testreal.txt', 'w', encoding = 'utf-8') as f:
        for label in test_labels:
            f.write(label + '\n')
    f.close()'''

    #建立词表
    # traindata = []
    # traindata = train_p[:]
    # traindata.extend(train_h)
    # build_vocab(traindata)

    #读取词表
    Vocab = read_vocab('data/Vocab.txt')
    vocab_size = len(Vocab)

    #生成embedding
    word_embedding = build_Wordembedding(Vocab)
    vocab_size = len(Vocab)

    #生成输入数据
    print(20*'=','准备训练......',20*'=')
    label_id = defaultdict(lambda:0)
    label_id['neutral'] = 0
    label_id['contradiction'] = 1
    label_id['entailment'] =2
    train_sen1,train_sen2,train_labels = create_input(train_p,train_h,train_labels,Vocab,label_id)
    vocab_size = len(Vocab)
    dev_sen1,dev_sen2,dev_labels = create_input(dev_p,dev_h,dev_labels,Vocab,label_id)
    vocab_size = len(Vocab)
    test_sen1,test_sen2,test_labels = create_input(test_p,test_h,test_labels,Vocab,label_id)


    vocab_size = len(Vocab)

    #训练&测试
    print(20*'=','开始训练......',20*'=')
    esim_model = ESIM_model(vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, CLASS_NUM, DROPOUT,word_embedding).to(device)
    optimizer = optim.Adam(esim_model.parameters(), lr=LEARNING_RATE, weight_decay = 1e-6) #, weight_decay = 1e-7
    loss_function = nn.CrossEntropyLoss(reduce=True,size_average=True)
    if TRAIN:
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
                batch_out = esim_model(batch_sen1.to(device), batch_sen2.to(device))

                loss = loss_function(batch_out.to(host),batch_label.squeeze(-1))  #计算损失值
                if hasattr(torch.cuda, 'empty_cache'):
	                torch.cuda.empty_cache()

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
                        batch_devout = esim_model(batch_devsen1.to(device), batch_devsen2.to(device))
                        batch_devlogits = torch.argmax(batch_devout.to(host), dim = 1)
                        pre_Y.extend(batch_devlogits.tolist())
                        real_Y.extend(batch_devlabel.tolist())
                    acc = accuracy_score(pre_Y, real_Y)
                    print('eval step %d, accuracy is %.4f' % (global_step, acc))
                    fd.write(str(acc) + '\n')
                    if acc > best_acc:
                        best_acc = acc
                        best_step = global_step
                        torch.save(esim_model,'best_model_ESIM.bin')
        print('All training finished. Best step is %d, best acc is %.4f' % (best_step, best_acc))
        fd.close()
        ft.close()
    else:
        # 模型测试
        print("开始测试......")
        model = torch.load('best_model_ESIM.bin')
        model.eval()
        test_dataloader = create_dataloader(test_sen1, test_sen2, shuffle = False)
        pre = []
        for batch_sen1,batch_sen2 in test_dataloader:
            batch_out = model(batch_sen1.to(device), batch_sen2.to(device))
            batch_logits = torch.argmax(batch_out.to(host), dim = 1)
            pre.extend(batch_logits.tolist())
        num_to_label = {0:'neutral',1:'contradiction',2:'entailment'}
        test = [num_to_label[i] for i in pre]
        with open('data/任务二-test.txt', 'w', encoding = 'utf-8') as f:
            for label in test:
                f.write(label + '\n')
        print("测试已完成......")
