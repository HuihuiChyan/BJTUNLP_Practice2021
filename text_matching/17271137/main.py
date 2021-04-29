import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchtext import data
from torchtext.vocab import Vectors
import argparse
import torch.nn.functional as F

parse = argparse.ArgumentParser()
parse.add_argument('-max_len', type=int, default=100)
parse.add_argument('-batch_size', type=int, default=64)
parse.add_argument('-embed_size', type=int, default=300)
parse.add_argument('-hidden_size', type=int, default=300)
parse.add_argument('-output_size', type=int, default=3)
parse.add_argument('-lr',type=float, default=0.0004)
parse.add_argument('-num_epochs', type=int, default=10)
args = parse.parse_args(args = [])
label_to_idx = {'neutral': 0, 'entailment': 1, 'contradiction': 2}

def get_dataiter(path, sequence, type='train'):
    labels, sentence1, sentence2 = [],[],[]
    label = data.Field(sequential=False, use_vocab=False)
    with open(path, 'r') as fp:
        dataset = fp.readlines()
    examples = []
    fields = [('seq_1',sequence),('seq_2',sequence),('label',label)]

    for idx in range(1,len(dataset)):
        label = dataset[idx].split('\t')[0]
        seq_1 = dataset[idx].split('\t')[5]
        seq_2 = dataset[idx].split('\t')[6]
        if label == '-':
            continue
        examples.append(data.Example.fromlist([seq_1, seq_2, label_to_idx[label]], fields))
    data_set = data.Dataset(examples, fields)
    if type == 'train':
        sequence.build_vocab(data_set, vectors=Vectors('glove.6B.300d.txt'))
        dataiter = data.BucketIterator(data_set, batch_size=args.batch_size, shuffle=True)
        return sequence, dataiter
    else:
        dataiter = data.BucketIterator(data_set, batch_size=args.batch_size, shuffle=False)
        return dataiter

def get_testiter(path, sequence, type):
    examples = []
    fields = [('seq_1',sequence),('seq_2',sequence)]
    with open(path,'r') as f:
        lines = f.readlines()
        for line in lines:
            seqs = line.strip().split('|||')
            examples.append(data.Example.fromlist(seqs,fields))
    test_dataset = data.Dataset(examples, fields)
    test_iter = data.Iterator(test_dataset, batch_size=args.batch_size, shuffle=False)
    return test_iter


class ESIM_model(nn.Module):
    def __init__(self, vocab):
        super(ESIM_model, self).__init__()
        self.embedding = nn.Embedding(len(vocab),args.embed_size)
        self.bilstm1 = nn.LSTM(args.embed_size, args.hidden_size, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(args.hidden_size, args.hidden_size, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(args.hidden_size*8, args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, args.output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

    def _attention(self, seq1_decoder, seq2_decoder, mask1, mask2):
        scores_b = torch.matmul(seq1_decoder, seq2_decoder.permute(0,2,1))
        scores_a = torch.matmul(seq2_decoder, seq1_decoder.permute(0,2,1))
        '''
        scores_b = scores_b.masked_fill(mask2.unsqueeze(-1) == 1, -1e9).transpose(1,2)
        scores_a = scores_a.masked_fill(mask1.unsqueeze(-1) == 1, -1e9).transpose(1,2)
        '''
        scores_b = F.softmax(scores_b, dim=-1)
        scores_a = F.softmax(scores_a, dim=-1)
        seq1_decoder_attention = torch.matmul(scores_b, seq2_decoder)
        seq2_decoder_attention = torch.matmul(scores_a, seq1_decoder)
        return seq1_decoder_attention,seq2_decoder_attention

    def _get_max_avg(self, x):
        p1 = F.avg_pool1d(x.transpose(1,2),x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1,2),x.size(1)).squeeze(-1)
        return torch.cat([p1,p2],dim=-1)

    def forward(self, seq1, seq2, mask1, mask2):
        seq1_embedding = self.embedding(seq1)
        seq2_embedding = self.embedding(seq2)

        seq1_decoder, _ = self.bilstm1(seq1_embedding)
        seq2_decoder, _ = self.bilstm2(seq2_embedding)
        seq1_decoder_attention, seq2_decoder_attention = self._attention(seq1_decoder, seq2_decoder, mask1, mask2)

        m_seq1 = torch.cat((seq1_decoder, seq1_decoder_attention, seq1_decoder-seq1_decoder_attention, torch.mul(seq1_decoder,seq1_decoder_attention)),dim=-1)
        m_seq2 = torch.cat((seq2_decoder, seq2_decoder_attention, seq2_decoder-seq2_decoder_attention, torch.mul(seq2_decoder,seq2_decoder_attention)),dim=-1)

        m_seq1_decoder = self.relu(self.linear1(m_seq1))
        m_seq2_decoder = self.relu(self.linear1(m_seq2))

        f_seq1_decoder, _ = self.bilstm2(m_seq1_decoder)
        f_seq2_decoder, _ = self.bilstm2(m_seq2_decoder)

        seq1_max_avg = self._get_max_avg(f_seq1_decoder)
        seq2_max_avg = self._get_max_avg(f_seq2_decoder)
        f_x = torch.cat([seq1_max_avg, seq2_max_avg],dim=-1)
        logit = self.linear2(self.dropout(self.tanh(self.linear1(f_x))))
        return logit

def val_test(net, data_iter, loss_fc):
    acc_sum, loss_sum, n, batch_count = 0.0, 0.0, 0, 0
    net.eval()
    for batch in data_iter:
        seq1 = batch.seq_1[0]
        seq2 = batch.seq_2[0]
        mask1 = (seq1==1)
        mask2 = (seq2==1)
        y = batch.label.cuda()
        y_hat = net(seq1.cuda(), seq2.cuda(), mask1.cuda(), mask2.cuda())
        acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
        n += y.shape[0]
        batch_count += 1
        loss_sum += loss_fc(y_hat, y).cpu().item()
    net.train()
    return acc_sum/n, loss_sum/batch_count

def train(net, train_iter, val_iter, test_iter, num_epochs, loss_fc, optimizer):
    net.train()
    min_num = 0.6
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for batch in train_iter:
            seq1 = batch.seq_1[0]
            seq2 = batch.seq_2[0]
            mask1 = (seq1==1)
            mask2 = (seq2==1)
            y = batch.label.cuda()
            y_hat = net(seq1.cuda(), seq2.cuda(), mask1.cuda(), mask2.cuda())
            l = loss_fc(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
            batch_count += 1
            n += y.shape[0]
            if batch_count%100 ==0:
                print('epoch: %d, train_loss: %.5f train_acc: %.5f'%(epoch, train_l_sum / batch_count, train_acc_sum /n))
            if batch_count%500 ==0:
                val_acc, val_num = val_test(net, val_iter, loss_fc)
                if val_acc>min_num:
                    min_num = val_acc
                    torch.save(net,'best_model3.bin')
                print('epoch: %d, train_loss: %.5f train_acc: %.5f val_loss: %.5f val_acc: %.5f'%(epoch, train_l_sum / batch_count, train_acc_sum /n, val_num, val_acc))

train_path = 'snli_1.0/snli_1.0_train.txt'
dev_path = 'snli_1.0/snli_1.0_dev.txt'
test_path = 'snli_1.0/snli.txt'
sequence = data.Field(sequential=True, use_vocab=True, lower=True, fix_length=args.max_len, include_lengths=True, batch_first=True)
sequence, train_iter = get_dataiter(train_path,sequence,type='train')
val_iter = get_dataiter(dev_path, sequence, type='val')
test_iter = get_testiter(test_path, sequence, type='test')

net = ESIM_model(sequence.vocab)
net.embedding.weight.data.copy_(sequence.vocab.vectors)
net = net.cuda()
loss_fc = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = args.lr, weight_decay=1e-6)
# train(net, train_iter, val_iter, test_iter, args.num_epochs, loss_fc, optimizer)

device = torch.device('cuda:0')
net.load_state_dict(torch.load('best_model.bin'))
net.to(device)

idx_to_label = {0:'neutral', 1:'entailment', 2:'contradiction'}

def test(net, data_iter):
    net.eval()
    res = []
    for batch in data_iter:
        seq1 = batch.seq_1[0]
        seq2 = batch.seq_2[0]
        mask1 = (seq1==1)
        mask2 = (seq2==1)
        y_hat = net(seq1.cuda(), seq2.cuda(), mask1.cuda(), mask2.cuda())
        predict = y_hat.argmax(dim=1).cpu()
        res += predict.tolist()
    with open('result.txt','w') as fp:
        for label in res:
            label = idx_to_label[label]
            fp.writelines(str(label)+'\n')

test(net, test_iter)