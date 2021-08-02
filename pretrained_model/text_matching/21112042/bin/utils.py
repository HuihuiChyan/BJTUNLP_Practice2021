import re

from torch.utils import data
import json

def load_data(tokenizer, batch_size, max_len):
    datapath = '/data/sunzengkui/datasets/snli/batch_'
    train_loader = prepro(datapath+'train.txt', tokenizer, batch_size, max_len, True)
    dev_loader = prepro(datapath+'dev.txt', tokenizer, batch_size, max_len, False)
    test_loader = prepro(datapath+'test.txt', tokenizer, batch_size, max_len, False)
    print('building dataset (train/dev/test) done.')
    return train_loader, dev_loader, test_loader

def load_infer(tokenizer, batch_size, max_len):
    datapath = '/data/sunzengkui/datasets/snli/'
    valid_loader = prepro(datapath + 'valid.txt', tokenizer, batch_size, max_len, False)
    print('building dataset (valid) done.')
    return valid_loader


def prepro(file_name, tokenizer, batch_size, max_len, shuffle=False):
    label_info = {
        "entailment": 0,
        "contradiction": 1,
        "neutral": 2,
    }
    data_list = []
    cnt = [0] * 3
    label_ = 0
    with open(file_name, 'r') as fin:
        for line in fin:
            cur_josn = json.loads(line.strip())
            if 'valid' not in file_name:
                label = cur_josn['gold_label']
                if label == '-':
                    label_ += 1
                    continue
                sent1 = regular(cur_josn['sentence1'])
                sent2 = regular(cur_josn['sentence2'])
                input_ids, input_mask = generate_ids(tokenizer, sent1, sent2, max_len)
                label = int(label_info[label])
                cnt[label] += 1
                sub_data = {'input_ids': input_ids,
                            'input_mask': input_mask,
                            'label': label}
            else:
                item = line.strip().split("|||")
                sent1 = regular(item[0].strip())
                sent2 = regular(item[1].strip())
                input_ids, input_mask = generate_ids(tokenizer, sent1, sent2, max_len)
                label = 1
                cnt[label] += 1
                sub_data = {'input_ids': input_ids,
                            'input_mask': input_mask,
                            'label': label}
            data_list.append(sub_data)
            # if len(data_list) == 100:
            #     break
    print("path:", file_name)
    print('data num:', len(data_list), ", '-' label num:", label_, ', label distribution:', cnt)
    dataset = Dataset(data_list)
    loader = data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=lambda x:x,
                             shuffle=shuffle)
    return loader

def regular(sentence):
    sentence = sentence.replace('.', ' . ').replace(',', ' , ').replace('?', ' ? ').replace('!', ' ! ')\
                        .replace('<', ' < ').replace('>', ' > ')
    sentence = re.sub(r"\s{2,}", " ", sentence)
    return sentence

def generate_ids(tokenizer, sentence1, sentence2, max_len):
    ids = tokenizer.encode(sentence1, sentence2)
    length = len(ids)
    mask = [1] * length
    if length < max_len:
        ids += [1] * (max_len - length)
        mask += [0] * (max_len - length)
    else:
        ids = ids[:max_len]
        mask = mask[:max_len]
    return ids, mask


def batch_know(sentences, tokenizer, max_len):
    sentences = sentences.split(" . </s> <s> ")
    know_ids, know_mask = [], []
    for sentence in sentences:
        ids = tokenizer.encode(sentence)
        mask = [1] * len(ids)
        klen = len(ids)
        if klen < max_len:
            ids += [1] * (max_len - klen)
            mask += [0] * (max_len - klen)
        else:
            ids = ids[:max_len]
            mask = mask[:max_len]
        know_ids.append(ids)
        know_mask.append(mask)
    return know_ids, know_mask

class Dataset(data.Dataset):
    def __init__(self, data_list):
        super(Dataset, self).__init__()
        self.data_list = data_list
        self.length = len(self.data_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data_list[idx]
