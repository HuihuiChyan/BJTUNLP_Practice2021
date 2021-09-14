import re
from collections import Counter, defaultdict
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
import pdb


def data_clean(neg_file, pos_file, neg_file_clean, pos_file_clean):
    with open(neg_file, 'r', encoding='utf-8')as neg_file, \
            open(pos_file, 'r', encoding='utf-8')as pos_file, \
            open(neg_file_clean, 'a', encoding='utf-8')as neg_file_clean, \
            open(pos_file_clean, 'a', encoding='utf-8')as pos_file_clean:
        for line in neg_file:
            line = re.sub(r'<br />', '', line)
            line = re.sub(r'!+', ' ! ', line)
            line = re.sub(r'\?+', ' ? ', line)
            line = re.sub(r'\.+', ' . ', line)
            line = re.sub(r',+', ' , ', line)
            line = re.sub(r'\(', ' ( ', line)
            line = re.sub(r'\)', ' ) ', line)
            line = re.sub(r'it\'s', 'it is', line)
            line = re.sub(r'i\'m', 'i am', line)
            line = re.sub(r'i\'ve', 'i have', line)
            line = re.sub(r'he\'s', 'he is', line)
            line = re.sub(r'that\'s', 'that is', line)
            line = re.sub(r'this\'s', 'this is', line)
            line = re.sub(r'can\'t', 'can not', line)
            line = re.sub(r'don\'t', 'do not', line)
            line = re.sub(r'does\'t', 'does not', line)
            line = re.sub(r'did\'t', 'did not', line)
            line = re.sub(r'isn\'t', 'is not', line)
            line = re.sub(r'[^a-zA-Z0-9(),!?\.\'\s]', '', line).lower()
            neg_file_clean.write(''.join(line))

        for line in pos_file:
            line = re.sub(r'<br />', '', line)
            line = re.sub(r'!+', ' ! ', line)
            line = re.sub(r'\?+', ' ? ', line)
            line = re.sub(r'\.+', ' . ', line)
            line = re.sub(r',+', ' , ', line)
            line = re.sub(r'\(', ' ( ', line)
            line = re.sub(r'\)', ' ) ', line)
            line = re.sub(r'it\'s', 'it is', line)
            line = re.sub(r'i\'m', 'i am', line)
            line = re.sub(r'i\'ve', 'i have', line)
            line = re.sub(r'he\'s', 'he is', line)
            line = re.sub(r'that\'s', 'that is', line)
            line = re.sub(r'this\'s', 'this is', line)
            line = re.sub(r'can\'t', 'can not', line)
            line = re.sub(r'don\'t', 'do not', line)
            line = re.sub(r'does\'t', 'does not', line)
            line = re.sub(r'did\'t', 'did not', line)
            line = re.sub(r'isn\'t', 'is not', line)
            line = re.sub(r'[^a-zA-Z0-9(),!?\.\'\s]', '', line).lower()
            pos_file_clean.write(''.join(line))


def build_vocab(neg_file_clean, pos_file_clean, vocab_file):
    with open(neg_file_clean, 'r', encoding='utf-8')as neg_file, \
            open(vocab_file, 'a', encoding='utf-8')as vocab_file, \
            open(pos_file_clean, 'r', encoding='utf-8')as pos_file:
        neg_lines = [line.split() for line in neg_file.readlines()]
        pos_lines = [line.split() for line in pos_file.readlines()]
        all_words = []
        for line in neg_lines:
            all_words.extend(line)
        for line in pos_lines:
            all_words.extend(line)

        counter = Counter(all_words)
        common_chars = counter.most_common()
        vocab_list = [pair[0] for pair in common_chars][:105000]  #107990
        for word in vocab_list:
            vocab_file.write(word + '\n')


def process_input(neg_file_clean, pos_file_clean, vocab_file, truncate=False):
    with open(neg_file_clean, 'r', encoding='utf-8')as neg_file, \
            open(vocab_file, 'r', encoding='utf-8')as vocab_file, \
            open(pos_file_clean, 'r', encoding='utf-8')as pos_file:
        vocab_lines = [line.strip() for line in vocab_file.readlines()]
        neg_lines = [line.strip() for line in neg_file.readlines()]
        pos_lines = [line.strip() for line in pos_file.readlines()]
        label_sentences = [0 for i in range(len(neg_lines))]
        label_sentences = label_sentences + [1 for i in range(len(pos_lines))]

        word_index = defaultdict(lambda: 0)
        for i, word in enumerate(vocab_lines):
            word_index[word] = i + 2

        all_sentences = []
        all_sentences.extend(neg_lines)
        all_sentences.extend(pos_lines)
        all_sentences = [[word_index[word] for word in line.split()] for line in all_sentences]

        len_all_sentences = [len(line) for line in all_sentences]

        if truncate:
            all_sentences = [line[:400] for line in all_sentences]
            all_sentences = [line + [1 for i in range(400 - len(line))] for line in all_sentences]
        else:
            max_len_sentences = max(len_all_sentences)
            all_sentences = [line + [1 for i in range(max_len_sentences - len(line))] for line in all_sentences]
        return all_sentences, label_sentences, len_all_sentences


def load_glove_file(file_in):
    embedding_index = {}
    with open(file_in, 'r', encoding='utf-8')as file_in:
        for line in file_in.readlines():
            line = line.split()
            word = line[0]
            embedding_index[word] = np.asarray(line[1:], dtype='float32')
    return embedding_index


embedding_size = 300


def build_embedding(file_vocab_in, glove_file):
    embedding_id = []
    embedding_index = load_glove_file(glove_file)
    embedding_id.append(np.asarray(np.random.uniform(-0.1, 0.1, embedding_size), dtype='float32'))
    embedding_id.append(np.asarray(np.random.uniform(-0.1, 0.1, embedding_size), dtype='float32'))
    id = 2
    with open(file_vocab_in, 'r', encoding='utf-8')as vocab_file:
        vocab_lines = [line.strip() for line in vocab_file.readlines()]
        for word in vocab_lines:
            if word in embedding_index.keys():
                embedding_id.append(embedding_index[word])
            else:
                embedding_id.append(np.asarray(np.random.uniform(-0.1, 0.1, embedding_size), dtype='float32'))
            id = id + 1
    return embedding_id


def create_dataloader(text_lines, label_lines, len_lines, shuffle=True):
    text_lines = torch.tensor(text_lines)  
    label_lines = torch.tensor(label_lines)
    len_lines = torch.tensor(len_lines)
    dataset = data.TensorDataset(text_lines, label_lines, len_lines)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=30,
                                 shuffle=shuffle)
    return dataloader


# vocab_size = 400000    # glove
vocab_size = 105002
class_num = 2
kernel_size = [3, 4, 5]
out_channels = 300
ci = 1
dropout = 0.5


embedding_id = build_embedding('vocab.txt', 'glove.6B.300d.txt')


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()

        # self.embedding = nn.Embedding(vocab_size, embedding_size)  
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_id))

        self.convs = nn.ModuleList([nn.Conv2d(ci, out_channels, (h, embedding_size))for h in kernel_size])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_channels*len(kernel_size), class_num)

    def forward(self, batched_text):
        embeds = self.embedding(batched_text)

        embeds = embeds.unsqueeze(1)

        relu_out = [F.relu(conv(embeds)).squeeze(3) for conv in self.convs]

        pool_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in relu_out]

        pool_out = torch.cat(pool_out, 1)

        dropout_out = self.dropout(pool_out)

        output = self.fc(dropout_out)

        return output


def cal_f1_acc(all_eval_logits, all_eval_labels):
    real_eval_logits = []  
    real_eval_labels = []  
    real_eval_logits.extend(all_eval_logits)
    real_eval_labels.extend(all_eval_labels)

    acc = accuracy_score(real_eval_labels, real_eval_logits)

    f1_value = f1_score(real_eval_labels, real_eval_logits, average='binary')

    return f1_value, acc


glove_embedding = load_glove_file('glove.6B.100d.txt')


train_sentences, train_sentences_label, train_sentences_len = \
    process_input('train_neg_clean_text.txt',
                  'train_pos_clean_text.txt',
                  'vocab.txt', truncate=True)
train_dataloader = create_dataloader(train_sentences, train_sentences_label, train_sentences_label, shuffle=True)


test_sentences, test_sentences_label, test_sentences_len = \
    process_input('test_neg_clean_text.txt',
                  'test_pos_clean_text.txt',
                  'vocab.txt')
test_dataloader = create_dataloader(test_sentences, test_sentences_label, test_sentences_label)


text_cnn = TextCNN().cuda()
optimizer = torch.optim.Adam(text_cnn.parameters(), lr=2e-4)
global_step = 0  
best_f1 = 0.0
best_accuracy = 0.0
best_step = 0
is_train = False


if is_train:
    for epoch in range(15):
        for batched_text, batched_label, batched_lens in train_dataloader:
            text_cnn.train()
            batched_output = text_cnn(batched_text.cuda())
            
            optimizer.zero_grad() 
            loss = torch.nn.functional.cross_entropy(batched_output.cpu(), batched_label)
            loss.backward()  
            optimizer.step()  
            
            global_step += 1

            if global_step % 50 == 0:
                print('train step %d, loss is %.4f' % (global_step, loss))

            if global_step % 250 == 0:
                text_cnn.eval()
                all_eval_logits = []
                all_eval_labels = []
                all_eval_lens = []
                for batched_text_test, batched_label_test, batched_lens_test in test_dataloader:
                    batched_output_test = text_cnn(batched_text_test.cuda())
                    batched_logits = torch.argmax(batched_output_test.cuda(), dim=-1)

                    all_eval_logits.extend(batched_logits.tolist())  
                    all_eval_labels.extend(batched_label_test.tolist())
                    all_eval_lens.extend(batched_lens_test.tolist())

                f1_value, acc = cal_f1_acc(all_eval_logits, all_eval_labels)
                
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_step = global_step
                    
                    best_model_name = r'TextCNN_glove_model_1' + '.bin'
                    torch.save(text_cnn, best_model_name)
                
                print('eval step %d, f1 value if %.4f, accuracy is %.4f' % (global_step, f1_value, acc))

                
        # print('All training finished.Best step is %d, best f1_value is %.4f' % (best_step, best_f1))
        print('All training finished.Best step is %d, best accuracy_value is %.4f' % (best_step, best_accuracy))
        
else:
    text_cnn = torch.load(r'TextCNN_glove_model_1.bin')
    text_cnn.eval()
    all_eval_logits = []
    all_eval_labels = []
    all_eval_lens = []
    for batched_text_dev, batched_label_dev, batched_lens_dev in test_dataloader:
        batched_output_dev = text_cnn(batched_text_dev.cuda())
        batched_logits = torch.argmax(batched_output_dev.cpu(), dim=-1)

        all_eval_logits.extend(batched_logits.tolist())
        all_eval_labels.extend(batched_label_dev.tolist())
        all_eval_lens.extend(batched_lens_dev.tolist())

    f1_value, acc = cal_f1_acc(all_eval_logits, all_eval_labels,)
    print('On our test set, f1 value if %.4f, accuracy is %.4f' % (f1_value, acc))
'''
    with open('test_clean.txt', 'r', encoding='utf-8')as test_file, \
            open('vocab.txt', 'r', encoding='utf-8')as vocab_file:
        vocab_lines = [line.strip() for line in vocab_file.readlines()]
        test_file = [line.strip() for line in test_file.readlines()]
        
        word_index = defaultdict(lambda: 0)
        for i, word in enumerate(vocab_lines):
            word_index[word] = i + 2

        all_sentences = []
        all_sentences.extend(test_file)
        all_sentences = [[word_index[word] for word in line.split()] for line in all_sentences]

        len_all_sentences = [len(line) for line in all_sentences]

        
        max_len_sentences = max(len_all_sentences)
        all_sentences = [line + [1 for i in range(max_len_sentences - len(line))] for line in all_sentences]
        
        text_lines = torch.tensor(all_sentences)
        len_all_sentences = torch.tensor(len_all_sentences)
        test_dataset = data.TensorDataset(text_lines, len_all_sentences)
        test_dataloader = data.DataLoader(dataset=test_dataset,
                                     batch_size=30,
                                     shuffle=False)
        
        text_cnn = torch.load(r'TextCNN_glove_model_1.bin')
        text_cnn.eval()
        
        with open('test_out.txt', 'w', encoding='utf-8')as test_out:
            all_eval_logits = []
            for batched_text, batched_len in test_dataloader:
                batched_output = text_cnn(batched_text.cuda())
                batched_logits = torch.argmax(batched_output.cpu(), dim=-1)
                all_eval_logits.extend(batched_logits.tolist()) 
            for logit in all_eval_logits:
                if logit == 0:
                    test_out.write('neg' + '\n')
                elif logit == 1:
                    test_out.write('pos' + '\n')
'''

        




