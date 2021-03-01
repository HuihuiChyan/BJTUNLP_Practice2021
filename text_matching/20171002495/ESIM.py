import torch
from torch import nn,optim
import torch.nn.functional as F

class ESIM(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 class_num,
                 dropout,
                 embeddings = None
                 ):
        super(ESIM, self).__init__()
        self.HiddenSize = hidden_size
        self.embedding_dim = embedding_dim
        self.LinearHiddenSize = hidden_size
        self.ClassNum = class_num
        self.dropout = dropout

        #embedding层
        self.embedding = nn.Embedding(vocab_size,self.embedding_dim)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.FloatTensor(embeddings))
        self.embedding.weight.requires_grad = False

        #BiLstm层（两个）
        self.bilstm1 = nn.LSTM(self.embedding_dim, self.HiddenSize, batch_first=True, dropout= self.dropout, bidirectional=True)
        self.bilstm2 = nn.LSTM(self.HiddenSize * 8, self.HiddenSize, batch_first=True, dropout= self.dropout, bidirectional=True)

        #全连接层
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.HiddenSize * 8, self.LinearHiddenSize),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.LinearHiddenSize, self.LinearHiddenSize),
            #nn.ELU(inplace=True),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.LinearHiddenSize, self.ClassNum),
            )
        #self.fc = nn.Linear(self.HiddenSize * 8, self.ClassNum)

    def attention_mechanism(self, x1, x2, mask1, mask2):  #注意力机制
        # batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))  #矩阵乘法
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)

        # x_atten: batch_size * seq_len * hidden_size
        x1_atten = torch.matmul(weight1, x2)
        x2_atten = torch.matmul(weight2, x1)

        return x1_atten, x2_atten

    def submul(self, x1, x2):   #计算差和点积
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def pooling(self, x):  #池化
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self,sen1_ids,sen2_ids,):

        #input encoding
        mask1, mask2 = sen1_ids.eq(0), sen2_ids.eq(0)
        sen1 = self.embedding(sen1_ids)
        sen2 = self.embedding(sen2_ids)    #batch_size * seq_len -> batch_size * seq_len * dim

        bilistm_1,_ = self.bilstm1(sen1)
        bilistm_2,_ = self.bilstm1(sen2)  # batch_size * seq_len * dim -> batch_size * seq_len * hidden_size


        #local inference modeling
        #注意力机制
        atten_1,atten_2 = self.attention_mechanism(bilistm_1, bilistm_2, mask1, mask2)
        # batch_size * seq_len * hidden_size   不改变维度

        #差和点积
        combined_1 = torch.cat([bilistm_1, atten_1, self.submul(bilistm_1, atten_1)], -1)
        combined_2 = torch.cat([bilistm_2, atten_2, self.submul(bilistm_2, atten_2)], -1)
        #batch_size * seq_len * hidden_size -> batch_size * seq_len * (8 * hidden_size)


        #inference composition
        bilistm_1, _ = self.bilstm2(combined_1)
        bilistm_2, _ = self.bilstm2(combined_2)  # batch_size * seq_len * (2 * hidden_size)
        #池化
        pool_1 = self.pooling(bilistm_1)
        pool_2 = self.pooling(bilistm_2)  # batch_size * (4 * hidden_size)
        # Classifier
        x = torch.cat([pool_1, pool_2], -1)
        similarity = self.fc(x)
        return similarity


