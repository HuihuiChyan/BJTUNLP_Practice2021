
from torch.utils import data
from bin.alphabet import Alphabet, regular
import json

def load_data(alphabet: Alphabet,
              batch_size: int,
              max_len: int):
    datapath = '/data/sunzengkui/datasets/snli/'
    train_loader = prepro(datapath+'train.txt', alphabet, batch_size, max_len, True)
    dev_loader = prepro(datapath+'dev.txt', alphabet, batch_size, max_len, False)
    test_loader = prepro(datapath+'test.txt', alphabet, batch_size, max_len, False)
    print('building dataset (train/dev/test) done.')
    return train_loader, dev_loader, test_loader

def load_infer(alphabet: Alphabet,
               batch_size: int,
               max_len: int):
    datapath = '/data/sunzengkui/datasets/snli/'
    valid_loader = prepro(datapath + 'valid.txt', alphabet, batch_size, max_len, False)
    print('building dataset (valid) done.')
    return valid_loader


def prepro(file_name, alphabet, batch_size, max_len, shuffle=False):
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
            if 'valid' not in file_name:
                cur_josn = json.loads(line.strip())
                label = cur_josn['gold_label']
                if label == '-':
                    label_ += 1
                    continue
                sent1 = regular(cur_josn['sentence1'])
                sent2 = regular(cur_josn['sentence2'])
                pre_ids, pre_mask = generate_ids(alphabet, sent1, max_len)
                hyp_ids, hyp_mask = generate_ids(alphabet, sent2, max_len)
                label = label_info[label]
                cnt[label] += 1
                sub_data = {'pre_ids': pre_ids,
                            'pre_mask': pre_mask,
                            'hyp_ids': hyp_ids,
                            'hyp_mask': hyp_mask,
                            'label': label}
            else:
                item = line.strip().split("|||")
                sent1 = regular(item[0].strip())
                sent2 = regular(item[1].strip())
                pre_ids, pre_mask = generate_ids(alphabet, sent1, max_len)
                hyp_ids, hyp_mask = generate_ids(alphabet, sent2, max_len)
                label = 1
                cnt[label] += 1
                sub_data = {'pre_ids': pre_ids,
                            'pre_mask': pre_mask,
                            'hyp_ids': hyp_ids,
                            'hyp_mask': hyp_mask,
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


def generate_ids(alphabet, sentence, max_len):
    ids = alphabet.encode(sentence)
    length = len(ids)
    mask = [1] * length
    if length < max_len:
        ids += [1] * (max_len - length)
        mask += [0] * (max_len - length)
    else:
        ids = ids[:max_len]
        mask = mask[:max_len]
    return ids, mask

class Dataset(data.Dataset):
    def __init__(self, data_list):
        super(Dataset, self).__init__()
        self.data_list = data_list
        self.length = len(self.data_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data_list[idx]
