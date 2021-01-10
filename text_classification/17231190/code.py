'''
只在动态的后面加了一层LSTM
'''
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

# ======================== ======================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--num_epoch', type=int, default=40)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--canshu', type=str, default='./model3.pt')
parser.add_argument('--acc_min', type=float, default=0.88)
parser.add_argument('--nums_channels', type=int, default=100)
args = parser.parse_args()
acc_min = args.acc_min
device = torch.device('cuda:1')

# ========================  ======================== #
with open('train.txt', encoding='utf-8') as ftrain:
    train_feature_line = [line.strip().split() for line in ftrain.readlines()]
train_label_line = [1] * (len(train_feature_line) // 2) + [0] * (len(train_feature_line) // 2)
with open('valid.txt', encoding='utf-8') as fvalid:
    valid_feature_line = [line.strip().split() for line in fvalid.readlines()]
valid_label_line = [1] * (len(valid_feature_line) // 2) + [0] * (len(valid_feature_line) // 2)

w2v_model = gensim.models.KeyedVectors.load_word2vec_format('glove2word2vec.6B.300d.txt', binary=False, encoding='utf-8')
print('loading glove_word2vec......')
word2id = dict(zip(w2v_model.wv.index2word, range(len(w2v_model.wv.index2word))))
id2word = {idx: word for idx, word in enumerate(w2v_model.wv.index2word)}
padding_value = word2id['pad']  # 10109
unk = word2id['unk']  # 201534


train_feature = [[word2id[word] if word in word2id else unk for word in line] for line in train_feature_line]
valid_feature = [[word2id[word] if word in word2id else unk for word in line] for line in valid_feature_line]


# 训练数据和标签，封装到列表中。
def get_dataset(features, labels):
    data = []
    for i in range(len(features)):
        temp = []
        temp.append(torch.tensor(features[i]).long())
        temp.append(torch.tensor([labels[i]]).long())
        data.append(temp)
    return data


train_data = get_dataset(train_feature, train_label_line)
test_dev_data = get_dataset(valid_feature, valid_label_line)

# print(len(train_data)) 25000
# print(len(test_data))  25000
# 验证集 5000 作为测试集。

test_data = []
dev_data = []

# dev
for x in range(len(test_dev_data) // 5):
    dev_data.extend(test_dev_data[5 * x: 5 * x + 4])

# test
for x in range(len(test_dev_data) // 5):
    test_data.extend(test_dev_data[5 * x + 4: 5 * x + 5])


def collate_fn(sample_data):
    sample_feature = []
    sample_label = []
    for data in sample_data:
        sample_feature.append(data[0])
        sample_label.append(data[1])
    sample_feature = rnn_utils.pad_sequence(sample_feature, batch_first=True, padding_value=padding_value)
    return sample_feature, sample_label


train_dataloader = DataLoader(train_data, args.batch_size, collate_fn=collate_fn, shuffle=True)
dev_dataloader = DataLoader(dev_data, args.batch_size, collate_fn=collate_fn, shuffle=True)
test_dataloader = DataLoader(test_data, args.batch_size, collate_fn=collate_fn, shuffle=True)


# 最后一个维度的最大池化。
class MaxPool(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])


class TextCNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embeding_vector, kernel_sizes, out_num_channels):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embeding_vector))
        self.embedding.weight.requires_grad = False
        self.constant_embedding = torch.nn.Embedding(input_size, hidden_size)
        self.constant_embedding.weight.data.copy_(torch.from_numpy(embeding_vector))

        self.dropout = torch.nn.Dropout(0.5)

        self.conv1 = torch.nn.Conv1d(2 * hidden_size, out_num_channels[0], kernel_sizes[0])
        self.conv2 = torch.nn.Conv1d(2 * hidden_size, out_num_channels[1], kernel_sizes[1])
        self.conv3 = torch.nn.Conv1d(2 * hidden_size, out_num_channels[2], kernel_sizes[2])

        self.linear = torch.nn.Linear(sum(out_num_channels), output_size)

        self.pool = MaxPool()

        self.biLSTM = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size//2, batch_first=True, dropout=0.5, bidirectional=True, num_layers=3)

    def forward(self, x):
        embedding1 = self.embedding(x)
        embedding2,hidden = self.biLSTM(embedding1)
        embeddings = torch.cat((embedding2, self.constant_embedding(x)), dim=2).permute(0, 2, 1)
        # [batch，len（seq)，2*embedding]      [batch，2*embedding，len（seq)]
        out1 = self.pool(F.relu(self.conv1(embeddings))).squeeze(-1)
        out2 = self.pool(F.relu(self.conv2(embeddings))).squeeze(-1)
        out3 = self.pool(F.relu(self.conv3(embeddings))).squeeze(-1)
        out = torch.cat((out1, out2, out3), dim=1)
        # conv(embeddings)  Conv1d(2*embedding, c, k)
        # [batch，c，由卷积核和步长共同决定]
        # pool
        # [batch，c，1]
        # squeeze(-1)
        # [batch，c]
        # cat
        # [batch，sum（out_num_channels）]
        out = self.linear(self.dropout(out))
        return out


loss_func = torch.nn.CrossEntropyLoss()
embedding_matrix = w2v_model.wv.vectors
input_size = embedding_matrix.shape[0]
hidden_size = embedding_matrix.shape[1]
kernel_size = [3, 4, 5]
out_nums_channels = [args.nums_channels, args.nums_channels, args.nums_channels]  # 设置输出通道数，相同即可。

model = TextCNN(input_size, hidden_size, 2, embedding_matrix, kernel_size, out_nums_channels).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if os.path.exists(args.canshu):
    print("loading model......")
    model.load_state_dict(torch.load(args.canshu))


def dev(model, dev_dataloader):
    test_loss, test_acc, n = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for data_x, data_y in dev_dataloader:
            label = torch.Tensor(data_y).long().to(device)
            out = model(data_x.to(device))
            loss = loss_func(out, label)
            prediction = out.argmax(dim=1).data.cpu().numpy()
            label = label.data.cpu().numpy()
            test_loss += loss.item()
            test_acc += accuracy_score(label, prediction)
            n += 1
    return test_loss / n, test_acc / n


def test(model, test_dataloader):
    test_loss, test_acc, n = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for data_x, data_y in test_dataloader:
            label = torch.Tensor(data_y).long().to(device)
            out = model(data_x.to(device))
            loss = loss_func(out, label)
            prediction = out.argmax(dim=1).data.cpu().numpy()
            label = label.data.cpu().numpy()
            test_loss += loss.item()
            test_acc += accuracy_score(label, prediction)
            n += 1
    return test_loss / n, test_acc / n

# train && dev
for epoch in range(args.num_epoch):
    model.train()
    train_loss, train_acc, n = 0.0, 0.0, 0
    for data_x, data_y in train_dataloader:
        label = torch.Tensor(data_y).long().to(device)
        out = model(data_x.to(device))
        # print(out.shape)  torch.Size([10, 2])
        # print(label.shape)  torch.Size([10])
        loss = loss_func(out, label)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prediction = out.argmax(dim=1).data.cpu().numpy()
        label = label.data.cpu().numpy()
        train_acc += accuracy_score(label, prediction)
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
model_test = TextCNN(input_size, hidden_size, 2, embedding_matrix, kernel_size, out_nums_channels).to(device)
if os.path.exists(args.canshu):
    print("loading model......")
    model_test.load_state_dict(torch.load(args.canshu))
test_loss, test_acc = test(model_test, test_dataloader)
print(('test_loss %f, test_accuracy %f' % (test_loss, test_acc)))
