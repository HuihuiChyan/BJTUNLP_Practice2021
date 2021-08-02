
from torch import nn
from torch.nn import functional as F
import torch

class ESIM_model(nn.Module):
    def __init__(self, config, vocab_len):
        super(ESIM_model, self).__init__()
        self.config = config
        self.embed = nn.Embedding(vocab_len, config['embed_size'])
        self.lstm1 = nn.LSTM(config['embed_size'], config['hidden_size'], bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(config['hidden_size'], config['hidden_size'], bidirectional=True, batch_first=True)
        self.linear1 = nn.Sequential(
            nn.Linear(config['hidden_size'] * 8, config['hidden_size']*2),
            nn.ReLU(),
            nn.Linear(config['hidden_size'] * 2, config['hidden_size'])
        )
        self.linear2 = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size'] // 2),
            nn.ReLU(),
            nn.Linear(config['hidden_size']//2, 3)
        )
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)


    def attention(self, seq1, mask1, seq2, mask2):
        # seq1: [batch, seqlen, embed_size]
        e1 = torch.bmm(seq2, seq1.permute(0, 2, 1))
        e2 = torch.bmm(seq1, seq2.permute(0, 2, 1))
        e1 = self.softmax1(e1)
        e2 = self.softmax2(e2)
        seq1_atten = torch.bmm(e2, seq2)
        seq2_atten = torch.bmm(e1, seq1)
        return seq1_atten, seq2_atten


    def pooling(self, seq):
        seq = seq.permute(0, 2, 1)
        max_seq = F.max_pool1d(seq, seq.shape[-1]).squeeze(dim=-1)
        avg_seq = F.avg_pool1d(seq, seq.shape[-1]).squeeze(dim=-1)
        return avg_seq, max_seq

    def forward(self, pre_ids, pre_mask, hyp_ids, hyp_mask, label):

        pre_ids = torch.tensor(pre_ids, dtype=torch.long).cuda()
        pre_mask = torch.tensor(pre_mask, dtype=torch.long).cuda()
        hyp_ids = torch.tensor(hyp_ids, dtype=torch.long).cuda()
        hyp_mask = torch.tensor(hyp_mask, dtype=torch.long).cuda()
        label = torch.tensor(label, dtype=torch.long).cuda()

        embed1, embed2 = self.embed(pre_ids), self.embed(hyp_ids)
        seq1_lstm, _ = self.lstm1(embed1)
        seq2_lstm, _ = self.lstm1(embed2)
        seq1_atten, seq2_atten = self.attention(seq1_lstm, pre_mask, seq2_lstm, hyp_mask)

        m1 = torch.cat((seq1_lstm, seq1_atten, seq1_lstm - seq1_atten, seq1_lstm.mul(seq1_atten)), dim=-1)
        m2 = torch.cat((seq2_lstm, seq2_atten, seq2_lstm - seq2_atten, seq2_lstm.mul(seq2_atten)), dim=-1)

        c1 = self.linear1(m1)
        c2 = self.linear1(m2)

        v1, _ = self.lstm2(c1)
        v2, _ = self.lstm2(c2)

        v1_avg, v1_max = self.pooling(v1)
        v2_avg, v2_max = self.pooling(v2)

        final_v = torch.cat((v1_avg, v1_max, v2_avg, v2_max), dim=-1)
        # print(final_v.shape)

        output = self.tanh(self.linear1(final_v))
        output = self.linear2(self.dropout(output))

        loss = F.cross_entropy(output, label)

        return loss, output.softmax(dim=-1)
