import collections
import os
import random
import re
import torch
from torch.utils import data as Data

from tqdm import tqdm

data_root = '/home/sunzengkui/Datasets/'

def file_pre(label):
    file_path = os.path.join(data_root, 'aclImdb', label)
    write_path = './data/'
    with open(os.path.join(write_path, label, 'feature.txt'), 'w', encoding='utf-8') as ff, \
            open(os.path.join(write_path, label, 'label.txt'), 'w', encoding='utf-8') as fl:
        for tag in ['pos', 'neg']:
            tag_path = os.path.join(file_path, tag)
            for file in tqdm(os.listdir(tag_path)):
                with open(os.path.join(tag_path, file), 'rb') as f:
                    review = f.read().decode('utf-8')
                    review = review.replace('\n', ' ').replace('\r', ' ')
                    ff.write(review + '\n')
                    fl.write(str(1 if tag == 'pos' else 0) + '\n')

def text_clean(text):
    text = text.replace('<br />', '')
    text = text.replace("'s", " is").replace("'m", " am").replace("'re", " are") \
                .replace("n't", " not").replace("'ll", " will").replace("'ve", " have")
    pattern = '\[\]\{\}\(\)'
    text = re.sub(r"[{}]".format(pattern), "", text.lower())
    text = re.sub(r"\.+", ' . ', text)
    text = re.sub(r",", ' , ', text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\"", " \" ", text)
    text = re.sub(r"\?", " ? ", text)
    # #
    text = re.sub(r"\s{2,}", ' ', text)
    return text.strip()

def file_read(label):
    file_path = './data/'
    data = []
    with open(os.path.join(file_path, label, 'feature.txt'), 'r', encoding='utf-8') as ff, \
            open(os.path.join(file_path, label, 'label.txt'), 'r', encoding='utf-8') as fl:
        f_reviews = ff.readlines()
        f_labels = fl.readlines()
        for review, tag in zip(f_reviews, f_labels):
            review = text_clean(review)
            data.append([review, int(tag)])

    print('load %s data file end and len = %d...' % (label, len(data)))

    random.shuffle(data)
    return data

# def glove2wv(glove_file_path, word2vec_file_path):


def build_vocab(glove_vocab):
    words = glove_vocab.itos
    word2idx = dict(zip(words, range(len(words))))
    idx2word = {idx: word for idx, word in enumerate(words)}
    return word2idx, idx2word

def get_loader(data, word2idx, batch_size, max_len):
    unk = word2idx['unk']
    pad = word2idx['pad']
    reviews, labels = [], []
    for item in data:
        reviews.append(item[0].split(' '))
        labels.append(item[1])

    labels = torch.tensor(labels)

    def Pad(X):
        return X[:max_len] if len(X) > max_len else X + [pad] * (max_len - len(X))

    features = torch.tensor([Pad([word2idx[word] if word in word2idx else unk for word in sentence]) for sentence in reviews])

    data_set = Data.TensorDataset(features, labels)
    data_iter = Data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # print(len(data_iter))
    return data_iter

def get_test_data(file_path, word2idx, max_len, batch_size):
    unk = word2idx['unk']
    pad = word2idx['pad']
    data = []

    def Pad(X):
        return X[:max_len] if len(X) > max_len else X + [pad] * (max_len - len(X))

    with open(file_path, 'r', encoding='utf-8') as f:
        test_lines = f.readlines()
        for line in test_lines:
            line = line.replace('\n', ' ').replace('\r', ' ')
            line = text_clean(line).split(' ')
            # tmp = Pad([word2idx[word] if word in word2idx else unk for word in line])
            data.append(Pad([word2idx[word] if word in word2idx else unk for word in line]))

    data_set = Data.TensorDataset(torch.tensor(data), torch.tensor([1] * len(data)))
    data_iter = Data.DataLoader(data_set, batch_size=batch_size, num_workers=4)

    return data_iter