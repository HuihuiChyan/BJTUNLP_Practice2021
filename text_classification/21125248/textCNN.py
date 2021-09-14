import pdb
import torch
import os
import numpy as np
from collections import Counter, defaultdict
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
import torchtext.vocab as Vocab
from DataProcessing import data_proc as test_data_proc

#Arguments
dataset_path = "."
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
embed_size = 300
batch_size = 64
filter_size = [2,3,4,5]
dropout_rate = 0.5
output_channel = 256
sen_len = 500

#切分数据
def data_proc(path):
    with open(path,'r', encoding='utf-8') as f:
        paras = f.readlines()
        datas = [para.strip('\n').split(' ') for para in paras]
    label = [data[0] for data in datas]
    sentences = [data[1:] for data in datas]
    return sentences, label

#加载数据
mergeFile_path = dataset_path + '/result.txt'
train_path = dataset_path + '/train' + '/train.txt'
dev_path = dataset_path + '/dev' + '/dev.txt'

mergeSentences, mergeLabels = data_proc(mergeFile_path)
trainSentences, trainLabels = data_proc(train_path)
devSentences, devLabels = data_proc(dev_path)

#词表
if not os.path.exists(dataset_path + '/vocab.txt'):
    train_words = [w for words in trainSentences for w in words]
    counter = Counter(train_words)
    common_words = counter.most_common()
    word_list = [words[0] for words in common_words[:63000]]
    word_list = ['<UNK>','<PAD>'] + word_list

    with open(dataset_path + '/vocab.txt', 'w', encoding='utf-8') as fv:
        for word in word_list:
            fv.write(word + '\n')
else:
    with open(dataset_path + '/vocab.txt', 'r', encoding='utf-8') as fv:
        word_list = [line for line in fv]

#glove
glove_vocab = Vocab.GloVe(name='6B', dim=300, cache="./glove/")
word2idx = dict(zip(glove_vocab.itos, range(len(glove_vocab.itos))))
idx2word = {idx: word for idx, word in enumerate(glove_vocab.itos)}
pad = word2idx['pad']  # 10109
unk = word2idx['unk']  # 201534
#print(pad,unk)


'''#unk和pad

word2idx = defaultdict(lambda:0)
idx2word = defaultdict(lambda:'unk')
for i,w in enumerate(word_list):
    word2idx[w.strip('\n')] = i
    idx2word[i] = w.strip('\n')
'''
vocab_size = len(word2idx)
num_classes = len(set(mergeLabels))  # 二分类问题

#pad补齐
max_len = sen_len

train_Sentences = [line[:max_len] for line in trainSentences]
dev_Sentences = [line[:max_len] for line in devSentences]

sens_train_pad = [sen + ['pad' for i in range(max_len - len(sen))] for sen in train_Sentences]
sens_val_pad = [sen + ['pad' for i in range(max_len - len(sen))] for sen in dev_Sentences]

#index转化
train_index = [[word2idx[word] if word in word2idx else unk for word in line] for line in sens_train_pad]
val_index = [[word2idx[word] if word in word2idx else unk for word in line] for line in sens_val_pad]

#label转化
train_label = [0 if label=='neg' else 1 for label in trainLabels]
val_label = [0 if label=='neg' else 1 for label in devLabels]

#装载数据
train_index = torch.LongTensor(train_index)
train_label = torch.LongTensor(train_label)
dataset_train = data.TensorDataset(train_index, train_label)
dataset_val = data.TensorDataset(torch.tensor(val_index), torch.tensor(val_label))
loader_train = data.DataLoader(dataset_train, batch_size, shuffle=True)
loader_val = data.DataLoader(dataset_val, batch_size, shuffle=False)

#构建模型
class textCNN(nn.Module):
    def __init__(self):
        super(textCNN, self).__init__()
        self.embedding_1 = nn.Embedding(vocab_size, embed_size)
        self.embedding_2 = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList(
            # 定义卷积
            # conv： [input_channel(=1), output_channel, (filter_height, filter_width), stride =1]
            [nn.Conv2d(1, output_channel, (k, embed_size * 2)) for k in filter_size]
            # 卷积以后的维度 [batch_size, output_channel, k, 1]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(output_channel * len(filter_size), num_classes)

    def forward(self, X):
        '''
        :param X: (batch_size, sequence_length)
        :return:
        '''
        batch_size = X[0]   #防止测试时维度不匹配
         #[batch_size, sequence_length,embed_size]
        #embedding_X = self.embedding_1(X)
        embedding_X = torch.cat((self.embedding_1(X), self.embedding_2(X)), dim=2)
        # 增加通道 [batch_size, channel(=1), sequence_length,embed_size*2]
        embedding_X = embedding_X.unsqueeze(1)
        conved = [F.relu(conv(embedding_X)).squeeze(3) for conv in self.convs]
        #pdb.set_trace()
        new_conved = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in conved]
            #通过完整convs之后的维度[batch_size, output_channel],期间顺便压缩了维度为1的空间
        output = torch.cat(new_conved, dim=1)
        output = self.fc(self.dropout(output))
        return output


#训练
def train(epoch_num, loader_train, loader_dev, model, crit, optim, device):
    best_acc = 0.0
    #training
    for epoch in range(epoch_num):
        model.train()
        total_loss, total_acc,  n = 0.0, 0.0, 0
        for X,y in loader_train:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            loss = crit(pred, y)

            optim.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optim.step()

            correct = (pred.argmax(dim=1) == y).float().sum().item()
            total_acc += correct
            total_loss += loss.item()
            n += y.shape[0]
        print('[Train]epoch %d, loss: %.4f, acc: %.4f' % (epoch + 1, total_loss/len(loader_train), total_acc/n))

        #validate per epoch
        model.eval()
        total_loss_d, total_acc_d, n = 0.0, 0.0, 0
        with torch.no_grad():
            for X, y in loader_dev:
                X, y = X.to(device), y.to(device)
                pred = model(X)

                total_acc_d += (pred.argmax(dim=1) == y).float().sum().item()
                n += y.shape[0]
            print('[Valid] acc: %.4f' % ( total_acc_d / n))

        # Save model
        if best_acc < (total_acc_d / n):
            best_acc = total_acc_d / n
            torch.save(model.state_dict(), os.path.join(dataset_path + '/ckpt', 'ckpt_best.pt'))

    print("\n\n\nBest acc on validation is : %.4f. \n\n\nTraining is Over!" % best_acc)

#测试
def test(loader_test, model, device):
    model.eval()
    with open(dataset_path + 'test_out.txt', 'w', encoding='utf-8') as f:
        with torch.no_grad():
            for X, y in loader_test:
                X = X.to(device)
                pred = model(X)
                pred = pred.argmax(dim=1).data.cpu().numpy()
                for tag in pred:
                    result_single = "neg" if tag == 0 else "pos"
                    f.write(result_single + "\n")

def get_loader_test():
    with open(dataset_path + '/test' + '/test.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [test_data_proc(line) for line in lines]
        testSentences = [line.strip('\n').split(' ') for line in lines]
    test_Sentences = [line[:max_len] for line in testSentences]
    sens_test_pad = [sen + ['pad' for i in range(max_len - len(sen))] for sen in test_Sentences]
    test_index = [[word2idx[word] if word in word2idx else unk for word in line] for line in sens_test_pad]
    # 装载数据
    dataset_test = data.TensorDataset(torch.tensor(test_index), torch.tensor([1] * len(test_index)))
    loader_test = data.DataLoader(dataset_test, batch_size, shuffle=False)
    return loader_test

def main():

    epoch_num = 30
    lr = 1e-4
    wd = 1e-7
    is_train = False

    if is_train:
        # 开始训练
        model = textCNN()
        model.embedding_1.weight.data.copy_(glove_vocab.vectors)
        model.embedding_2.weight.data.copy_(glove_vocab.vectors)
        model.embedding_2.requires_grad_(False)
        model = model.to(device)

        # 优化器与交叉损失函数
        crit = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        train(epoch_num, loader_train, loader_val, model, crit, optim, device)
    else:
        model = textCNN()
        # 加载模型
        model_name = 'ckpt_best.pt'
        checkpoint = torch.load(os.path.join(dataset_path + '/ckpt', model_name))
        model.load_state_dict(checkpoint)
        model = model.to(device)

        loader_test = get_loader_test()

        # 开始预测
        test(loader_test, model, device)

if __name__ == '__main__':
    main()





