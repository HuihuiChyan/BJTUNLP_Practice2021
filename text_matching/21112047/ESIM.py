import re
import torch
import pdb
import numpy as np
import json
from torch import nn
from torch.utils import data
from collections import Counter, defaultdict
from sklearn.metrics import f1_score, accuracy_score

def load_data(file_in, file_out):
    with open(file_in, 'r', encoding='utf-8')as file_in, \
            open(file_out, 'w', encoding='utf-8')as file_out:
        lines = [re.sub('\s\(.*\)', '', line.strip()).lower() for line in file_in.readlines()]
        lines = lines[1:]
        for line in lines:
            # line = re.sub('\d{5}.*', '', line)
            file_out.write(''.join(line) + '\n')

def separate_texts_label_and_clean(file_in, file_out_text1, file_out_text2, file_out_label):
    label_lines = []
    text_1_lines = []
    text_2_lines = []
    with open(file_in, 'r', encoding='utf-8')as file_in, \
            open(file_out_text1, 'w', encoding='utf-8')as file_out_text1, \
            open(file_out_text2, 'w', encoding='utf-8')as file_out_text2, \
            open(file_out_label, 'w', encoding='utf-8')as file_out_label:
        lines = [line.strip().split('\t') for line in file_in.readlines()]
        for line in lines:
            if len(line) > 2 and len(line[0]) in [7, 10, 13]:
                label_lines.append(line[0])
                '''
                line[1] = re.sub(r'[\.\"\!\?\,]', ' ', line[1])
                line[2] = re.sub(r'[\.\"\!\?\,]', ' ', line[2])
                line[1] = re.sub(r'\'s', '', line[1])
                line[2] = re.sub(r'\'s', '', line[2])
                line[1] = re.sub(r'\'t', '', line[1])
                line[2] = re.sub(r'\'t', '', line[2])
                '''
                text_1_lines.append(line[1])
                text_2_lines.append(line[2])
                file_out_text1.write(''.join(line[1]) + '\n')
                file_out_text2.write(''.join(line[2]) + '\n')
                file_out_label.write(''.join(line[0]) + '\n')

    #return text_1_lines, text_2_lines, label_lines


load_data('train.txt', 'train_texts_and_label.txt')
load_data('dev.txt', 'dev_texts_and_label.txt')


separate_texts_label_and_clean('train_texts_and_label.txt', 'train_text1.txt', 'train_text2.txt', 'train_label.txt')

separate_texts_label_and_clean('dev_texts_and_label.txt', 'dev_text1.txt', 'dev_text2.txt', 'dev_label.txt')

def build_vocabulary(file_text_1, file_text_2, file_out):
    with open(file_out, 'w', encoding='utf-8')as file_out, \
            open(file_text_1, 'r', encoding='utf-8')as file_text_1, \
            open(file_text_2, 'r', encoding='utf-8')as file_text_2:
        text_1_lines = [line.split() for line in file_text_1.readlines()]
        text_2_lines = [line.split() for line in file_text_2.readlines()]
        all_words = []
        for line in text_1_lines:
            all_words.extend(line)
        for line in text_2_lines:
            all_words.extend(line)

        words_counter = Counter(all_words)
        common_words = words_counter.most_common()
        print(len(common_words))
        vocabulary_list = [pair[0] for pair in common_words]

        for words in vocabulary_list[:50000]:
            file_out.write(''.join(words) + '\n')

build_vocabulary('train_text1.txt', 'train_text2.txt', 'vocabulary.txt')

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def load_glove_file(file_in):
    embedding_index = {}
    with open(file_in, 'r', encoding='utf-8')as glove_file:
        for line in glove_file.readlines():
            line = line.split()
            if is_number(line[1]):
                word = line[0]
                embedding_index[word] = np.asarray(line[1:], dtype='float32')

    return embedding_index


embedding_size = 300

def build_embedding(glove_file):
    embedding_id = []
    embedding_index = load_glove_file(glove_file)
    #embedding_id.append(np.asarray(np.random.randn(embedding_size), dtype='float32'))
    #embedding_id.append(np.asarray(np.random.randn(embedding_size), dtype='float32'))
    embedding_id.append(embedding_index['UNK'])
    embedding_id.append(embedding_index['PAD'])
    with open('vocabulary.txt', 'r', encoding='utf-8')as vocab_file:
        vocab_lines = [line.strip() for line in vocab_file.readlines()]
        for word in vocab_lines:
            if word in embedding_index.keys() and len(embedding_index[word]) == 300:
                embedding_id.append(embedding_index[word])
            else:
                #embedding_id.append(embedding_index['UNK'])
                embedding_id.append(np.asarray(np.random.randn(embedding_size), dtype='float32'))
    return embedding_id

embedding_id = build_embedding('/data/wangpanpan/task2/glove.840B.300d.txt')

def process_input(file_text1, file_text2, file_label, file_vocabulary, truncate=False):
    with open(file_text1, 'r', encoding='utf-8') as file_text1, \
            open(file_text2, 'r', encoding='utf-8') as file_text2, \
            open(file_label, 'r', encoding='utf-8') as file_label, \
            open(file_vocabulary, 'r', encoding='utf-8') as file_vocabulary:
        vocabulary_lines = [vocab.strip() for vocab in file_vocabulary.readlines()]

        char2idx = defaultdict(lambda: 0)
        for i, char in enumerate(vocabulary_lines):
            char2idx[char] = i + 2

        tag2idx = {'neutral': 0, 'contradiction': 1, 'entailment': 2}
        label_lines = [line.strip() for line in file_label.readlines()]
        label_lines = [tag2idx[line] for line in label_lines]

        text1_lines = [list(line.strip()) for line in file_text1.readlines()]
        text1_lines = [[char2idx[word] for word in line] for line in text1_lines]

        text2_lines = [list(line.strip()) for line in file_text2.readlines()]
        text2_lines = [[char2idx[word] for word in line] for line in text2_lines]

        if truncate:
            text1_lines = [line[:96] for line in text1_lines]
            text2_lines = [line[:96] for line in text2_lines]

        len_lines = [len(line) for line in text1_lines]
        len_lines = len_lines + [len(line) for line in text2_lines]
        max_len = max(len_lines)

        text1_lines = [line + [1 for i in range(max_len - len(line))] for line in text1_lines]
        text2_lines = [line + [1 for i in range(max_len - len(line))] for line in text2_lines]

        return text1_lines, text2_lines, label_lines

train_text_1_lines, train_text_2_lines, train_label_lines = \
    process_input('train_text1.txt', 'train_text2.txt', 'train_label.txt', 'vocabulary.txt', truncate=True)

dev_text_1_lines, dev_text_2_lines, dev_label_lines = \
    process_input('dev_text1.txt', 'dev_text2.txt', 'dev_label.txt', 'vocabulary.txt')

def create_dataloader(text1_lines, text2_lines, label_lines, shuffle=False, drop_last=False):
    text1_lines = torch.tensor(text1_lines)
    text2_lines = torch.tensor(text2_lines)
    label_lines = torch.tensor(label_lines)
    dataset = data.TensorDataset(text1_lines, text2_lines, label_lines)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=200,
                                 shuffle=shuffle,
                                 drop_last=drop_last)
    return dataloader

train_dataloader = create_dataloader(train_text_1_lines, train_text_2_lines, train_label_lines, shuffle=True, drop_last=True)
dev_dataloader = create_dataloader(dev_text_1_lines, dev_text_2_lines, dev_label_lines)

hidden_size = 300
linear_size = 300
class ESIM(nn.Module):
    def __init__(self):
        super(ESIM, self).__init__()

        #self.embeds = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.embeds = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_id))
        self.lstm1 = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)

        self.linear1 = nn.Sequential(
            nn.Linear(hidden_size * 8, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, 3)
        )
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        
    def soft_attention_align(self, x1, x2, mask1, mask2):
        """
        :param x1: batch_size * seq_len * hidden_size
        :param x2: batch_size * seq_len * hidden_size
        """

        # attention: batch_size * seq_len * seq_len

        attention1 = torch.matmul(x1, x2.transpose(1, 2))
        attention2 = torch.matmul(x2, x1.transpose(1, 2))

        # weight: batch_size * seq_len * seq_len
        weight1 = torch.nn.functional.softmax(attention1 + mask1.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = torch.nn.functional.softmax(attention2 + mask2.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)

        # x_align: batch_size * seq_len * hidden_size
        return x1_align, x2_align


    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        """
        :param x: batch_size * seq_len * (2 * hidden_size)
        """
        p1 = torch.nn.functional.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = torch.nn.functional.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)

        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, text1, text2):
        """
        :param text1: batch_size * seq_len
        :param text2: batch_size * seq_len
        """
        mask1, mask2 = text1.eq(1), text2.eq(1)

        # embeds: batch_size * seq_len => batch_size * seq_len * embeds_dim
        x1 = self.embeds(text1)
        x2 = self.embeds(text2)

        # batch_size * seq_len * embeds_dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)

        # q1_align, q2_align: batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)
        
        q1 = self.linear1(q1_combined)
        q2 = self.linear1(q2_combined)

        # q1_compose, q2_compose: batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1)
        q2_compose, _ = self.lstm2(q2)

        # q1_rep, q2_rep: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        out = torch.cat([q1_rep, q2_rep], -1)
        
        out = self.tanh(self.linear1(out))
        out = self.linear2(out)
        return out

esim_model = ESIM().cuda()
optim = torch.optim.Adam(esim_model.parameters(), lr=6e-4, weight_decay=0.)
global_step = 0
best_acc = 0.0
best_step = 0
is_train = True

if is_train:
    for epoch in range(50):
        for batched_text1, batched_text2, batched_label in train_dataloader:
            esim_model.train()
            batched_output = esim_model(batched_text1.cuda(), batched_text2.cuda())
            optim.zero_grad()
            loss = torch.nn.functional.cross_entropy(batched_output.cpu(), batched_label)
            loss.backward()
            optim.step()

            if global_step % 100 == 0:
                print('train step %d, loss is %.4f' % (global_step, loss))

            if global_step % 1000 == 0:
                # evaluation!
                esim_model.eval()
                all_eval_logits = []
                all_eval_labels = []
                for batched_text1, batched_text2, batched_label in dev_dataloader:
                    batched_output = esim_model(batched_text1.cuda(), batched_text2.cuda())
                    batched_logits = torch.argmax(batched_output.cuda(), dim=-1)

                    all_eval_logits.extend(batched_logits.tolist())
                    all_eval_labels.extend(batched_label.tolist())
                acc = accuracy_score(all_eval_labels, all_eval_logits)

                print('eval step %d, accuracy is %.4f' % (global_step, acc))

                if acc > best_acc:
                    best_acc = acc
                    best_step = global_step

                    best_model_name = r'best_model_1.bin'
                    torch.save(esim_model, best_model_name)
            global_step += 1
    print('All training finished.Best step is %d, best accuracy is %.4f' % (best_step, best_acc))

else:
    esim_model = torch.load(r'best_model_1.bin')
    esim_model.eval()
    all_eval_logits = []
    all_eval_labels = []
    for batched_text1, batched_text2, batched_label in test_dataloader:
        batched_output = esim_model(batched_text1.cuda(), batched_text2.cuda())
        batched_logits = torch.argmax(batched_output.cpu(), dim=-1)

        all_eval_logits.extend(batched_logits.tolist())
        all_eval_labels.extend(batched_label.tolist())

    acc = accuracy_score(all_eval_labels, all_eval_logits)

    print('Test accuracy is %.4f' % (acc))