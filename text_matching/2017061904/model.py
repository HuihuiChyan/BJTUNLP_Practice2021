import torch
import torch.nn as nn
import torch.nn.functional as F

class ESIM(nn.Module):
    def __init__(self, args):

        super(ESIM, self).__init__()
        self.vocab_size = args.vocab_size
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.target_size = args.target_size

        # Embedding
        self.Embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.Embedding.weight.data.copy_(args.weight_matrix)
        self.Embedding.weight.requires_grad = False
        
        self.lstm1 = nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=0.5, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_dim * 8, self.hidden_dim, dropout=0.5, batch_first=True, bidirectional=True)

        #self.fc = nn.Linear(self.hidden_dim * 8, self.target_size)
        self.f = nn.Sequential(
            nn.Linear(self.hidden_dim * 8, self.hidden_dim * 8),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * 8, self.hidden_dim * 8),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * 8, self.hidden_dim * 8),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * 8),
			nn.Linear(self.hidden_dim * 8, 200),
			nn.ELU(inplace=True),
			nn.BatchNorm1d(200),
			nn.Dropout(0.5),
			nn.Linear(200, 200),
			nn.ELU(inplace=True),
			nn.BatchNorm1d(200),
			nn.Dropout(0.5),
			nn.Linear(200, 4)
        )


    def soft_attention_align(self, data1, data2):
        # batch_size * len1 * len2
        attention = torch.matmul(data1, data2.transpose(1, 2))
        # batch_size * len1 * len2
        weight1 = F.softmax(attention, dim=-1)
        # batch_size * len2 * len1
        weight2 = F.softmax(attention.transpose(1, 2), dim=-1)

        data1 = torch.matmul(weight1, data2)
        data2 = torch.matmul(weight2, data1)

        return data1, data2

    def submul(self, in1, in2):
        sub = in1 - in2
        mul = in1 * in2
        return torch.cat([sub, mul], -1)

    def get_pooled(self, x):
        max = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        avg = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        return torch.cat([max, avg], -1)

    def forward(self, data1, data2):
        # batch_size * len -> batch_size * len * embedding_dim    
        data1 = self.Embedding(data1)
        data2 = self.Embedding(data2)
        # batch_size * len * embedding_dim -> batch_size * len * hidden_dim    
        data1, _ = self.lstm1(data1)
        data2, _ = self.lstm1(data2)

        align1, align2 = self.soft_attention_align(data1, data2)
        
        concat1 = torch.cat([data1, align1, self.submul(data1, align1)], -1)
        concat2 = torch.cat([data2, align2, self.submul(data2, align2)], -1)

        result1, _ = self.lstm2(self.f(concat1))
        result2, _ = self.lstm2(self.f(concat2))
        
        final1 = self.get_pooled(result1)
        final2 = self.get_pooled(result2)
        
        final = torch.cat([final1, final2], -1)
        final = self.fc(final)

        return final



