import os
import pdb
import re

import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset
import torchtext.vocab as Vocab

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

#目录
data_dir = "."
dataset_dir = "snli_1.0"

#glove
glove_vocab = Vocab.GloVe(name='6B', dim=300, cache="./glove/")
word2idx = dict(zip(glove_vocab.itos, range(len(glove_vocab.itos))))
idx2word = {idx: word for idx, word in enumerate(glove_vocab.itos)}

# 设置标签集合
label_set = {"neutral": 0, "entailment": 1, "contradiction": 2}

#部分参数
vocab_size = len(word2idx)
num_classes = len(label_set)  # 多少种标签类型
seq_len = 128
batch_size = 256

def text_proc(text):
    text = text.lower()

    text = text.replace("?", " ? ").replace(".", " . ").replace(",", " , ")\
        .replace("!", " ! ").replace(";", " ; ")

    text = text.replace("&", " and ").replace("'s", " is").replace("'m", " am") \
        .replace("'re", " are").replace("won't", "will not").replace("can't", "can not") \
        .replace("n't", " not").replace("'ve", " have").replace("'ll", " will")

    r = r"[_+=——$%^~@#￥%&*《》<>「」{}【】/]"
    text = re.sub(r, " ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text

def get_data(is_train = True, is_test = False):
    if is_test:
        file_path = os.path.join(data_dir, dataset_dir, "snli.test") #"snli.test"

        with open(file_path, "r") as f:
            rows = [line.split("|||") for line in f.readlines()]
        premises = [text_proc(row[0])for row in rows]
        hypothesis = [text_proc(row[1]) for row in rows]
        labels = [ 0 for _ in rows]

    else:
        file_path = os.path.join(data_dir, dataset_dir, "snli_1.0_train.txt" if is_train else "snli_1.0_dev.txt")
        with open(file_path, "r") as f:
            rows = [line.split("\t") for line in f.readlines()[1:]]
        premises = [text_proc(row[5]) for row in rows if row[0] in label_set]
        hypothesis = [text_proc(row[6]) for row in rows if row[0] in label_set]
        labels = [label_set[row[0]] for row in rows if row[0] in label_set]

    return premises, hypothesis, labels

def get_token_pad(lines):

    pad = word2idx['pad']  # 10109
    unk = word2idx['unk']  # 201534

    # pad补齐
    data_pad = [data[:seq_len] for data in lines]
    data_pad = [data + ['pad' for i in range(seq_len - len(data))] for data in data_pad]

    # index转化
    data_index = np.array([[word2idx[word] if word in word2idx else unk for word in line] for line in data_pad])

    return data_index

def get_iter(batch_size):

    #获取数据
    train_data = get_data()
    dev_data = get_data(is_train=False)
    test_data = get_data(is_test=True)

    # 装载数据
    dataset_train = MyDataset(train_data)
    dataset_dev = MyDataset(dev_data)
    dataset_test = MyDataset(test_data)

    loader_train = data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    loader_dev = data.DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=False)
    loader_test = data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    return loader_train, loader_dev, loader_test

class My_Dataset(Dataset):
    def __init__(self, data):
        super(Dataset, self).__init__()
        self.data = data
        self.len = len(self.data)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx], self.data[2][idx]

class MyDataset(data.Dataset):
    def __init__(self, dataset):
        pres = [premise.strip().split(' ') for premise in dataset[0]]
        hypos = [hypothesis.strip().split(' ') for hypothesis in dataset[1]]

        self.premises = get_token_pad(pres)
        self.hypothesis = get_token_pad(hypos)
        self.labels = np.array(dataset[2])

        print('read' + str(len(self.premises)) + 'examples')

    def __getitem__(self, idx):
        return self.premises[idx], self.hypothesis[idx], self.labels[idx]

    def __len__(self):
        return len(self.premises)

#模型
class ESIM(nn.Module):
    def __init__(self):
        super(ESIM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = 300
        self.hidden_size = 300
        self.dropout = 0.5
        self.num_class = num_classes

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.copy_(glove_vocab.vectors)

        self.biLSTM1 = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_size * 8, self.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2)
        )

        self.biLSTM2 = nn.LSTM(self.hidden_size * 2, self.hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, self.num_class)
        )

    def soft_align_attention(self, a_hat, b_hat):
        attention1 = torch.matmul(a_hat, b_hat.transpose(1, 2))

        weight1 = F.softmax(attention1, dim=-1)
        weight2 = F.softmax(attention1.transpose(1, 2), dim=-1)

        a_atten = torch.matmul(weight1, b_hat)
        b_atten = torch.matmul(weight2, a_hat)

        return a_atten, b_atten

    def avg_max_pool(self, v):

        v_avg = F.avg_pool1d(v.transpose(1,2), v.size(1)).squeeze(-1)
        v_max = F.max_pool1d(v.transpose(1,2), v.size(1)).squeeze(-1)

        return torch.cat([v_avg, v_max], dim =-1)


    def forward(self, a, b):

        #a,b: (batch_size, seq_len) → a,b: (batch_size, seq_len, embedding_size)
        a = self.embedding(a)
        b = self.embedding(b)

        # a_hat,b_hat: (batch_size, seq_len, embedding_size) → (batch_size, seq_len, 2*hidden_size)
        a_hat, _ = self.biLSTM1(a)
        b_hat, _ = self.biLSTM1(b)

        #Attention
        #（batch_size,seq_len,hidden_size * 2）
        a_atten, b_atten = self.soft_align_attention(a_hat, b_hat)

        # （batch_size,seq_len,hidden_size * 2） → （batch_size,seq_len,hidden_size * 8）
        a_con = torch.cat([a_hat, a_atten, a_hat - a_atten, a_hat.mul(a_atten)], dim = -1)
        b_con = torch.cat([b_hat, b_atten, b_hat - b_atten, b_hat.mul(b_atten)], dim = -1)

        a_con = self.linear1(a_con)
        b_con = self.linear1(b_con)

        # a_hat,b_hat: (batch_size, seq_len, hidden_size * 2) → (batch_size, seq_len, hidden_size)
        v_a, _ = self.biLSTM2(a_con)
        v_b, _ = self.biLSTM2(b_con)

        v = torch.cat([self.avg_max_pool(v_a), self.avg_max_pool(v_b)], dim=-1)

        y = self.fc(self.linear1(v))
        return y


def train(train_iter, dev_iter, model, epoch_num, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Training Start!\n")
    best_acc = 0.
    for epoch in range(epoch_num):
        n, loss_num = 0, 0.0
        for i, (x1, x2, y) in enumerate(train_iter):
            model.train()
            x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
            y_hat = model(x1, x2)
            l = F.cross_entropy(y_hat, y)

            loss_num += l.item()

            #梯度
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            n += y.shape[0]

            if i % 100 == 0:
                print('[TRAIN] Epoch: {:>3d} Batch: {:4d} Loss: {:.4f}'.format(
                    epoch + 1, (i + 1) * batch_size, l.item()))
            if i % 500 == 0:
                # 验证
                dev_acc = dev(dev_iter, model)

                # 存储模型
                if best_acc < dev_acc:
                    best_acc = dev_acc
                    torch.save(model.state_dict(), os.path.join('./ckpt', 'ckpt_best.pt'))

        print('[Train]epoch %d is over, avg loss: %.4f' % (epoch + 1, loss_num / n))

    print("Training is Over! \n\n\nThe best acc on Dev is %.4f." % best_acc)


def dev(dev_iter, model):
    model.eval()
    acc_sum_d, n_d = 0., 0
    with torch.no_grad():
        for x1, x2, y in dev_iter:
            x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
            y_hat = model(x1, x2)
            # print(y_hat,y)
            # print(y_hat.argmax(dim=1))
            acc_sum_d += (y_hat.argmax(dim=1) == y).float().sum().item()
            n_d += y.shape[0]
        print('[Valid] acc: %.4f' % (acc_sum_d / n_d))
    model.train()
    return acc_sum_d / n_d


#测试
def test(test_iter, model):
    model.eval()
    with open(data_dir + 'test_out.txt', 'w', encoding='utf-8') as f:
        total_preds = []
        id_label = {0: 'neutral', 1: 'entailment', 2: 'contradiction'}
        with torch.no_grad():
            for x1, x2, _ in test_iter:
                x1, x2 = x1.cuda(), x2.cuda()
                y_hat = model(x1, x2)

                total_preds += list(y_hat.argmax(dim=1).cpu().detach().numpy())
            for tag in total_preds:
                f.write(id_label[tag] + "\n")

            print('[test] Over!')

def main():
    train_iter, dev_iter, test_iter = get_iter(batch_size=batch_size)
    is_train = False
    epoch_num = 50
    lr = 1e-3
    if is_train:
        model = ESIM()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.cuda()
        train(train_iter, dev_iter, model, epoch_num, lr)
    else:
        # 加载模型
        model_name = 'ckpt_best.pt'
        checkpoint = torch.load(os.path.join('./ckpt', model_name))
        model = ESIM()
        model.load_state_dict(checkpoint)
        model = model.cuda()
        # 开始预测
        test(test_iter, model)


if __name__ == '__main__':
    main()



