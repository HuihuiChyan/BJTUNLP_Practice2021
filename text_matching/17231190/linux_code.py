from torch import nn
import datetime
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from gensim.models import Word2Vec
import argparse
from sklearn.metrics import accuracy_score
import os
import gensim
from sacremoses import MosesTokenizer

# ======================================= 一些初始化参数 ======================================= #
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0004)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--canshu', type=str, default='model_canshu1.pt')
parser.add_argument('--acc_min', type=float, default=0.66)
args = parser.parse_args()

device = torch.device('cuda:1')
batch_size = args.batch_size
acc_min = args.acc_min
canshu = args.canshu
weight_decay = args.weight_decay
lr = args.lr
num_epoch = args.num_epoch

# ======================================= 加载数据集并处理 ======================================= #
# 训练集
mt = MosesTokenizer(lang='en')
with open('./snli_1.0/train_sentence1_split.txt', 'r', encoding='utf-8') as f:
    train_feature1_line = [mt.tokenize(line.strip.lower(), return_str=True) for line in f.readlines()]
with open('./snli_1.0/train_sentence2_split.txt', 'r', encoding='utf-8') as f:
    train_feature2_line = [mt.tokenize(line.strip.lower(), return_str=True) for line in f.readlines()]
with open('./snli_1.0/train_gold_label.txt', 'r', encoding='utf-8') as f:
    train_label_line = [line.strip() for line in f.readlines()]
# 验证集
with open('./snli_1.0/dev_sentence1_split.txt', 'r', encoding='utf-8') as f:
    dev_feature1_line = [mt.tokenize(line.strip.lower(), return_str=True) for line in f.readlines()]
with open('./snli_1.0/dev_sentence2_split.txt', 'r', encoding='utf-8') as f:
    dev_feature2_line = [mt.tokenize(line.strip.lower(), return_str=True) for line in f.readlines()]
with open('./snli_1.0/dev_gold_label.txt', 'r', encoding='utf-8') as f:
    dev_label_line = [line.strip() for line in f.readlines()]
# 测试集
with open('./snli_1.0/test_sentence1_split.txt', 'r', encoding='utf-8') as f:
    test_feature1_line = [mt.tokenize(line.strip.lower(), return_str=True) for line in f.readlines()]
with open('./snli_1.0/test_sentence2_split.txt', 'r', encoding='utf-8') as f:
    test_feature2_line = [mt.tokenize(line.strip.lower(), return_str=True) for line in f.readlines()]
with open('./snli_1.0/test_gold_label.txt', 'r', encoding='utf-8') as f:
    test_label_line = [line.strip() for line in f.readlines()]

# split
train_feature1_line = [line.split(" ") for line in train_feature1_line]
train_feature2_line = [line.split(" ") for line in train_feature2_line]
dev_feature1_line = [line.split(" ") for line in dev_feature1_line]
dev_feature2_line = [line.split(" ") for line in dev_feature2_line]
test_feature1_line = [line.split(" ") for line in test_feature1_line]
test_feature2_line = [line.split(" ") for line in test_feature2_line]

# 加载训练好的词向量：
# 使用的是glove，但是glove中只有unk，pad不敢确定。
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('glove2word2vec.6B.300d.txt', binary=False, encoding='utf-8')
print('loading word2vec......')
word2id = dict(zip(w2v_model.wv.index2word, range(len(w2v_model.wv.index2word))))
id2word = {idx: word for idx, word in enumerate(w2v_model.wv.index2word)}
feature_pad = word2id['pad']  # 10109
feature_unk = word2id['unk']  # 201534
label2id = {'neutral': 0, 'entailment': 1, 'contradiction': 2}  # 可以确定模型最后一层
label_pad = 0

# token转化为下标：
train_feature1 = [[word2id[word] if word in word2id else feature_unk for word in words] for words in train_feature1_line]
train_feature2 = [[word2id[word] if word in word2id else feature_unk for word in line] for line in train_feature2_line]
train_label = [[label2id[word] if word in label2id else label_pad] for word in train_label_line]
dev_feature1 = [[word2id[word] if word in word2id else feature_unk for word in line] for line in dev_feature1_line]
dev_feature2 = [[word2id[word] if word in word2id else feature_unk for word in line] for line in dev_feature2_line]
dev_label = [[label2id[word] if word in label2id else label_pad] for word in dev_label_line]
test_feature1 = [[word2id[word] if word in word2id else feature_unk for word in line] for line in test_feature1_line]
test_feature2 = [[word2id[word] if word in word2id else feature_unk for word in line] for line in test_feature2_line]
test_label = [[label2id[word] if word in label2id else label_pad] for word in test_label_line]


# ======================================= torchText ======================================= #

# 训练数据和标签，封装到列表中。
def get_dataset(feature1, feature2, labels):
    data = []
    for i in range(len(feature1)):
        temp = []
        temp.append(torch.tensor(feature1[i]).long())
        temp.append(torch.tensor(feature2[i]).long())
        temp.append(torch.tensor([labels[i]]).long())
        data.append(temp)
    return data


train_data = get_dataset(train_feature1, train_feature2, train_label)
test_data = get_dataset(test_feature1, test_feature2, test_label)
dev_data = get_dataset(dev_feature1, dev_feature2, dev_label)


def collate_fn(sample_data):
    sample_feature1 = []
    sample_feature2 = []
    sample_label = []
    for data in sample_data:
        sample_feature1.append(data[0])
        sample_feature2.append(data[1])
        sample_label.append(data[2])
    sample_feature1 = rnn_utils.pad_sequence(sample_feature1, batch_first=True, padding_value=feature_pad)
    sample_feature2 = rnn_utils.pad_sequence(sample_feature2, batch_first=True, padding_value=feature_pad)

    return sample_feature1, sample_feature2, sample_label


train_dataloader = DataLoader(train_data, args.batch_size, collate_fn=collate_fn, shuffle=True)
dev_dataloader = DataLoader(dev_data, args.batch_size, collate_fn=collate_fn, shuffle=True)
test_dataloader = DataLoader(test_data, args.batch_size, collate_fn=collate_fn, shuffle=True)



# ======================================= ESIM模型核心部分 ======================================= #
class ESIM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_vector):
        # 总词汇的个数， 嵌入的深度， 3分类问题的3， 具体的每一个词的词向量组成
        super(ESIM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_vector))
        self.embedding.weight.requires_grad = False

        self.bilstm1 = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, dropout=0.5, bidirectional=True)
        # （ 50, 50 ）
        self.bilstm2 = torch.nn.LSTM(input_size=hidden_size * 8, hidden_size=hidden_size, batch_first=True, dropout=0.5, bidirectional=True)
        # （ 400, 50    2（双向） * 4（增强时4个拼接）

        self.linear1 = torch.nn.Linear(hidden_size * 8, 3)
        self.drop = torch.nn.Dropout(0.3)

    def attention(self, seq1, seq2):
        eik = torch.matmul(seq2, seq1.transpose(1, 2))
        ekj = torch.matmul(seq1, seq2.transpose(1, 2))
        # (10, 25, 25）

        eik = F.softmax(eik, dim=-1)
        ekj = F.softmax(ekj, dim=-1)
        # [10, 25, 25]

        ai = torch.matmul(ekj, seq2)  # torch.Size([10, 25, 25])  *  torch.Size([10, 25, 100])  因为双向 LSTM 50*2 了
        bj = torch.matmul(eik, seq1)
        # torch.Size([10, 25, 100])
        return ai, bj


    def forward(self, seq1, seq2):
        seq1 = self.embedding(seq1)
        seq2 = self.embedding(seq2)
        # （10, 25，50）
        bilstm_1, hidden1 = self.bilstm1(seq1)
        bilstm_2, hidden2 = self.bilstm1(seq2)
        #（10，25，100）
        ai, bj = self.attention(bilstm_1, bilstm_2)
        #（10，25，100）    注意力机制，不改变维度。
        # 论文中的四部分
        ma = torch.cat([bilstm_1, ai, bilstm_1-ai, bilstm_1*ai], -1)
        mb = torch.cat([bilstm_2, bj, bilstm_2-bj, bilstm_2*bj], -1)
        # （10，25，100 + 100 + 100 + 100 = 8 * 50 = 400）
        bilstm_1, hidden1 = self.bilstm2(ma)  # （ 400 ， 50 ）
        bilstm_2, hidden2 = self.bilstm2(mb)
        # （ 10，25，800 ）
        # 池化：
        p1 = F.avg_pool1d(bilstm_1.transpose(1, 2), bilstm_1.size(1)).squeeze(-1)
        p2 = F.max_pool1d(bilstm_1.transpose(1, 2), bilstm_1.size(1)).squeeze(-1)
        p3 = F.avg_pool1d(bilstm_2.transpose(1, 2), bilstm_2.size(1)).squeeze(-1)
        p4 = F.max_pool1d(bilstm_2.transpose(1, 2), bilstm_2.size(1)).squeeze(-1)
        # 拼接
        output = torch.cat([p1, p2, p3, p4], -1)
        # torch.Size([10, 400])
        # 3分类
        output = self.linear1(self.drop(output))

        return output


# ======================================= train ======================================= #
embedding_matrix = w2v_model.wv.vectors
input_size, hidden_size = embedding_matrix.shape[0], embedding_matrix.shape[1]
loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数 ，
model = ESIM(input_size, hidden_size, len(label2id), embedding_matrix).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=argsweight_decay)
if os.path.exists(args.canshu):
    print("loading model......")
    model.load_state_dict(torch.load(args.canshu))


def dev(model, dev_dataloader):
    test_loss, test_acc, n = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for seq1, seq2, label in dev_dataloader:
            model.train()
            out = model(seq1.to(device), seq2.to(device))
            loss = loss_func(out, label.squeeze(-1))
            prediction = out.argmax(dim=1).data.cpu().numpy().tolist()
            label = label.view(1, -1).squeeze().data.cpu().numpy().tolist()
            train_acc += accuracy_score(label, prediction)
            train_loss += loss.item()
            n += 1
    return test_loss / n, test_acc / n


def test(model, test_dataloader):
    test_loss, test_acc, n = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for seq1, seq2, label in test_dataloader:
            model.train()
            out = model(seq1.to(device), seq2.to(device))
            loss = loss_func(out, label.squeeze(-1))
            prediction = out.argmax(dim=1).data.cpu().numpy().tolist()
            label = label.view(1, -1).squeeze().data.cpu().numpy().tolist()
            train_acc += accuracy_score(label, prediction)
            train_loss += loss.item()
            n += 1
    return test_loss / n, test_acc / n


# train && dev
for epoch in range(args.num_epoch):
    train_loss, train_acc, n = 0.0, 0.0, 0
    for seq1, seq2, label in train_dataloader:
        # torch.Size([10, 25])
        # torch.Size([10, 25])  批量为10，每个句子25个单词。
        # torch.Size([10, 1])   每个句子对应的标签。
        model.train()
        out = model(seq1.to(device), seq2.to(device))
        # print(out.shape)   [10, 3]   （批量，三分类问题）
        loss = loss_func(out, label.squeeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prediction = out.argmax(dim=1).data.cpu().numpy().tolist()
        label = label.view(1, -1).squeeze().data.cpu().numpy().tolist()
        train_acc += accuracy_score(label, prediction)
        train_loss += loss.item()
        n += 1

    dev_loss, dev_acc = dev(model, dev_dataloader)
    print('epoch %d, train_loss %f, train_accuracy %f, dev_loss %f, dev_accuracy %f' %
            (epoch + 1, train_loss/n, train_acc/n, dev_loss, dev_acc))
    if dev_acc > acc_min:
        acc_min = dev_acc
        torch.save(model.state_dict(), args.canshu)
        print("save model...")

# test
print("test ...")
model_test = ESIM(input_size, hidden_size, len(label2id), embedding_matrix).to(device)
if os.path.exists(args.canshu):
    print("loading model......")
    model_test.load_state_dict(torch.load(args.canshu))
test_loss, test_acc = test(model_test, test_dataloader)
print(('test_loss %f, test_accuracy %f' % (test_loss, test_acc)))






'''
# (2,3,4)
eik = torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]],

                    [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]
                    ])
print(eik.shape)

# （2,3）
mask1 = torch.tensor([[1, 0, 1],

                       [1, 0, 1]
                        ])

eik = eik.masked_fill(mask1.unsqueeze(-1) == 1, -1e9)
print(eik)

输出：

torch.Size([2, 3, 4])
tensor([[[-1000000000, -1000000000, -1000000000, -1000000000],
         [          1,           1,           0,           0],
         [-1000000000, -1000000000, -1000000000, -1000000000]],

        [[-1000000000, -1000000000, -1000000000, -1000000000],
         [          1,           1,           0,           0],
         [-1000000000, -1000000000, -1000000000, -1000000000]]])
'''