
from torch import nn
from torch.nn import functional as F
import torch

class BERT_model(nn.Module):
    def __init__(self, config, bert_model):
        super(BERT_model, self).__init__()
        self.bert = bert_model
        self.config = config
        self.fc = nn.Sequential(nn.Tanh(), nn.Linear((config['hidden']), config['classes']))

    def forward(self, input_ids, input_mask, label):
        input_ids = torch.tensor(input_ids, dtype=torch.long).cuda()
        input_mask = torch.tensor(input_mask, dtype=torch.long).cuda()
        label = torch.tensor(label, dtype=torch.long).cuda()

        print('input ids shape:', input_ids.shape)
        print('input mask shape:', input_mask.shape)

        text_output = self.bert(input_ids=input_ids, attention_mask=input_mask)
        print('text output shape:', text_output[0].shape)
        batch, seqlen, hidden = text_output[0].shape

        # know_output = know_output[0][:, ]
        text_output = text_output[0][:, 0:1, :].squeeze(dim=1)
        # know_output = know_output[0][:, 0:1, :]
        print('text output shape:', text_output.shape)

        final = text_output.view([batch, -1])

        preds = self.fc(final)
        loss = F.cross_entropy(preds, label)
        return loss, preds
