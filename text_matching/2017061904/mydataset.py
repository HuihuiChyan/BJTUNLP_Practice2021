import os
import re
import json

import torchtext.data as data

class DS(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, data_path, test=False):
        def clean_str(str):
            str = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", str)
            str = re.sub(r"\'s", " \'s", str)
            str = re.sub(r"\'ve", " \'ve", str)
            str = re.sub(r"n\'t", " n\'t", str)
            str = re.sub(r"\'re", " \'re", str)
            str = re.sub(r"\'d", " \'d", str)
            str = re.sub(r"\'ll", " \'ll", str)
            str = re.sub(r"\'", " \' ", str)
            str = re.sub(r"\"", " \" ", str)
            str = re.sub(r",", " , ", str)
            str = re.sub(r"!", " ! ", str)
            str = re.sub(r"\(", " ( ", str)
            str = re.sub(r"\)", " ) ", str)
            str = re.sub(r"\?", " ? ", str)

            return str.strip()
        text_field.tokenize = lambda x: clean_str(x).split()
        '''
        fields = {"sentence1" : text_field,
                  "sentence2" : text_field, 
                  "gold_label" : label_field}
        '''
        fields = [("sentence1", text_field),
                  ("sentence2", text_field),
                  ("label", label_field)]
        #fields = [("text", text_field), ("label", label_field)]

        examples = self.get_examlples(data_path, text_field, label_field, fields, test)

        super(DS, self).__init__(examples, fields)

    def get_examlples(self, data_path, text_field, label_field, fields, test=False):
        if test is True:
            with open(data_path, "r", encoding="UTF-8") as f:
                lines = [line.split("|||") for line in f.readlines()]
                for i in range(len(lines)):
                    lines[i][0] = lines[i][0].strip()
                    lines[i][1] = lines[i][1].strip()
                    lines[i].append("neutral")
                examples = [data.Example.fromlist(dt, fields) for dt in lines]
        else:
            fields = {"sentence1" : ("sentence1", text_field),
                  "sentence2" : ("sentence2", text_field), 
                  "gold_label" : ("label", label_field)}
            with open(data_path, "r", encoding="UTF-8") as f:
                lines = [line.strip() for line in f.readlines()]
                examples = [data.Example.fromJSON(dt, fields) for dt in lines]
                #print(len(examples))
                #examples = examples[:int(len(examples)*0.1)]
        
        return examples
