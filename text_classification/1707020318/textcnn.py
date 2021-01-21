import torch
from torch import nn
import torch.nn.functional as F

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])


class TextCNN(nn.Module):
    def __init__(self, vocab_len, embedding_size, kernel_sizes, num_channels, dropout_p):
        """
        :param vocab: 词典
        :param embedding_size: embedding 层向量维度
        :param kernel_sizes: 卷积层 filter size
        :param num_channels: 输出通道数
        :param dropout_p: dropout 参数
        """
        super(TextCNN, self).__init__()

        self.bilstm_embedding = nn.Embedding(vocab_len, embedding_size)
        self.const_embedding = nn.Embedding(vocab_len, embedding_size)

        self.dropout = nn.Dropout(dropout_p)

        self.pool = GlobalMaxPool1d()
        self.bilstm = nn.LSTM(embedding_size, embedding_size//2,
                              bidirectional=True,
                              num_layers=3)
        self.conv_list = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.conv_list.append(nn.Conv1d(in_channels=3*embedding_size,
                                            out_channels=c,
                                            kernel_size=k))
        self.decoder = nn.Linear(sum(num_channels), 2)

    def forward(self, X):
        """
        :param X: [batch_size, seq_len]
        :return:
        """
        # embedding output: torch([batch_size, seq_len, embedding_size])
        # BiLSTM input: torch([seq_len, batch_size, hidden_size])
        # BiLSTM output: torch([seq_len, batch_size, num_directions * hidden_size])
        embedding1 = self.bilstm_embedding(X)
        embedding2, _ = self.bilstm(embedding1)
        embedding2 = embedding2
        embeddings = torch.cat((embedding1, embedding2, self.const_embedding(X)), dim=2)

        # embeddings: torch([batch_size, embedding_size * 2, seq_len])
        embeddings = embeddings.permute(0, 2, 1)

        # Conv1d input: [batch_size, channels, seq_len]
        # Conv1d output: [batch_size, channels, seq_len - kernel_size + 1]
        # Pool output: [batch_size, channels, 1]
        # encoding output: [batch_size, sum(channels)]
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.conv_list], dim=1)

        outputs = self.decoder(self.dropout(encoding))
        return outputs