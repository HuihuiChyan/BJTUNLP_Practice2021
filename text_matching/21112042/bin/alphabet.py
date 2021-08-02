
import collections
import json
import re
import torch
import torchtext.vocab as Vocab
import numpy as np


class Alphabet:
    def __init__(self):
        self.instance2idx = collections.defaultdict(lambda :0)
        self.instances = []
        self.next_idx = 0

        self.unk, self.pad = 0, 1
        self.add_instance('unk')
        self.add_instance('pad')

    def add_instance(self, instance):
        instance = instance.lower()
        if instance not in self.instance2idx:
            self.instance2idx[instance] = self.next_idx
            self.instances.append(instance)
            self.next_idx += 1

    def get_idx(self, instance):
        if instance in self.instance2idx:
            return self.instance2idx[instance]
        else:
            try:
                return self.instance2idx[instance.lower()]
            except:
                return self.unk

    def encode(self, sentence):
        sentence = sentence.split()
        ids = []
        for word in sentence:
            ids.append(self.get_idx(word))
        return ids

    def save(self, path='../alphabet.json'):
        save_json = {'dict': self.instance2idx,
                     'list': self.instances,
                     'index': self.next_idx}
        try:
            json.dump(save_json, open(path, 'w'))
            print(f'successfully write {self.next_idx} words.')
        except Exception as e:
            print(f'write words error, e = {e}.')

    def load(self, path='../alphabet.json'):
        try:
            cur_json = json.load(open(path))
            self.instance2idx = cur_json['dict']
            self.instances = cur_json['list']
            self.next_idx = cur_json['index']
            print(f'successfully load {self.next_idx} words.')
        except Exception as e:
            print(f'load words error, e = {e}.')

    def process_glove_vectors(self, path='./vocab_embed.txt'):
        glove_vocab = Vocab.GloVe(name='840B', dim=300, cache='/data/sunzengkui/datasets/glove')
        embedding_matrix = np.empty((self.next_idx, 300))
        for word, ids in self.instance2idx.items():
            try:
                idx = glove_vocab.stoi[word]
            except:
                idx = glove_vocab.stoi['unk']
            finally:
                embed = glove_vocab.vectors[idx].numpy()
            embedding_matrix[ids] = embed.tolist()

        try:
            np.savetxt(path, embedding_matrix)
            print('pretrain embedding saved.')
        except:
            print('save pretrain embedding error.')

    def load_vocab(self, path='./bin/vocab_embed.txt'):
        embed_matrix = np.loadtxt(path)
        return torch.tensor(embed_matrix, dtype=torch.float32)




def regular(sentence):
    sentence = sentence.replace('.', ' . ').replace(',', ' , ').replace('?', ' ? ').replace('!', ' ! ')\
                        .replace('<', ' < ').replace('>', ' > ')
    sentence = re.sub(r"\s{2,}", " ", sentence)
    return sentence



if __name__ == '__main__':
    data_base = '/data/sunzengkui/datasets/snli/'
    alpha = Alphabet()
    with open(data_base + 'train.txt') as fin:
        for line in fin:
            cur_json = json.loads(line.strip())
            sent1 = regular(cur_json['sentence1'].strip()).split()
            sent2 = regular(cur_json['sentence2'].strip()).split()
            for word in sent1:
                alpha.add_instance(word)
            for word in sent2:
                alpha.add_instance(word)
    alpha.save()
    alpha.process_glove_vectors()