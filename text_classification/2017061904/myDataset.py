import re
import os
import random

import pandas as pd

import torch
from torchtext import data

root_path = os.path.abspath('.')
model_path = os.path.join(root_path, 'model')
data_path = os.path.join(root_path, 'data')
train_data_path = os.path.join(data_path, 'train')
test_data_path = os.path.join(data_path, 'test')

features = []
targets = []

class DS(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, data_path, text_field, label_field, shuffle=False, test=False):
        def clean_str(str):
            str = str.replace("<br />", "", 10000)
            str = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", str)
            str = re.sub(r"\'s", " \'s", str)
            str = re.sub(r"\'ve", " \'ve", str)
            str = re.sub(r"n\'t", " n\'t", str)
            str = re.sub(r"\'re", " \'re", str)
            str = re.sub(r"\'d", " \'d", str)
            str = re.sub(r"\'ll", " \'ll", str)
            str = re.sub(r",", " , ", str)
            str = re.sub(r"!", " ! ", str)
            str = re.sub(r"\(", " ( ", str)
            str = re.sub(r"\)", " ) ", str)
            str = re.sub(r"\?", " ? ", str)
            str = re.sub(r"\s{2,}", " ", str)

            return str.strip()

        fields = [("text", text_field), ("label", label_field)]
        self.datas = []
        if(test):
            self.load_data(data_path)
            examples = [data.Example.fromlist(dt, fields) for dt in self.datas]
        else:
            text_field.tokenize = lambda x: clean_str(x).split()
            self.load_data(data_path, "pos")
            self.load_data(data_path, "neg")
            if shuffle:
                random.shuffle(self.datas)
            
            examples = [data.Example.fromlist(dt, fields) for dt in self.datas]

        super(DS, self).__init__(examples, fields)

    def load_data(self, file_path, select=None):
        if select is not None:
            file_path = os.path.join(file_path, select)
            files = os.listdir(file_path)
            for file in files:
                path = os.path.join(file_path, file)
                with open(path, 'r', encoding="UTF-8") as f:
                    n = len(file)
                    line = []
                    str = f.read()
                    line.append(str)
                    label = int(file[:n-4].split('_')[1])
                    if(label > 5):
                        label = 'pos'
                    else:
                        label = 'neg'
                    line.append(label)
                    self.datas.append(line)
        else:
            with open(file_path, "r", encoding="UTF-8") as f:
                lines = [line.strip() for line in f.readlines()]
                for i in lines:
                    line = []
                    line.append(i)
                    line.append("1")
                    self.datas.append(line)
                
