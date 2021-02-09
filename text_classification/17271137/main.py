import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.utils import data
from gensim.test.utils import datapath
import os
import re
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import logging
import jieba
from sklearn.metrics import accuracy_score
import datetime

class MyDataset(Dataset):
    # Pytorch中有工具函数torch.utils.Data.DataLoader，通过这个函数我们在准备加载数据集使用mini-batch的时候可以使用多线程并行处理，
    # 这样可以加快我们准备数据集的速度。Datasets就是构建这个工具函数的实例参数之一。
    # 自己定义的dataset类需要继承pytorch的Dataset
    # 需要实现必要的魔法方法:
    # – __init__魔法方法里面进行读取数据文件
    # – __getitem__魔法方法进行支持下标访问
    # – __len__魔法方法返回自定义数据集的大小，方便后期遍历
    # 一般__init__负责加载全部原始数据，初始化之类的。__getitem__负责按索引取出某个数据，并对该数据做预处理。
    # 但是对于如何加载原始数据以及如何预处理数据完全是由自己定义的，包括我们用 dataset[index] 取出的数据的组织形式都是完全自行定义的。
    def __init__(self,file_list,label_list,sentence_max_size,embedding,word2id,stopwords):
        self.x = file_list
        self.y = label_list
        self.sentence_max_size = sentence_max_size
        self.embedding = embedding
        self.word2id = word2id
        self.stopwords = stopwords
    def __getitem__(self,index):
        # 使用dataloader时，需要得到（返回）的是评论的tensor矩阵 和 评论的类别
        words=[]
        with open(self.x[index],"r",encoding='utf-8') as file:
            for line in file.readlines():
                words.extend(segment(line.strip(),stopwords))
        tensor = generate_tensor(words,self.sentence_max_size,self.embedding,self.word2id)
        return tensor,self.y[index]
    def __len__(self):
        return len(self.x)
'''
以“文章”为单位.
len是文章的个数，__getitem__返回的也是整个文章的词向量矩阵（以文章为单位进行的词典构造并生成词向量）。
'''

def read_testdata(test_dir):
    
    with open(test_dir, "r", encoding='utf-8') as file:
        print(datetime.datetime.now())
        print('句子处理：')
        textlines = [segment(line.strip(),stopwords) for line in file.readlines()]
        print(datetime.datetime.now())
        print('生成tensor：')
        text = torch.cat([generate_tensor(line,sentence_max_size,embedding,word2id) for line in textlines],0)
        
    return text.unsqueeze(1)
            

class TextCNN(nn.Module):
    def __init__(self, vec_dim, filter_num, sentence_max_size, label_size, kernel_list):
        super(TextCNN,self).__init__()
        channel_num = 1
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channel_num,filter_num,(kernel,vec_dim)),
            nn.ReLU(),
            nn.MaxPool2d((sentence_max_size-kernel+1,1))
        )
            for kernel in kernel_list])
        # 对应每个kernel，有filter_num个。
        # out_channel = filter_num
        # kernel的维度：考虑kernel与矩阵的乘法，kernel_size * vec_dim(与矩阵乘法对应)
        # 因此共有 filter_num * len(kernel_list) 个，对应linear的参数。
        self.fc = nn.Linear(filter_num*len(kernel_list),label_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,x):
        # conv2d的输入[batch_size, channel, sentence_max_size, vec_dim]
        in_size = x.size(0) # batch_size
        out = [conv(x) for conv in self.convs]
        # conv(x)卷积并最大池化，得到输出维度：[batch_size, filter_num(out_channel), 1, 1]
        # out是一个list，对3种size的kernel作了遍历，out包含 len(kernel_list)=3 个：[batch_size, filter_num, 1, 1]的tensor
        out = torch.cat(out,dim=1) 
        # 将每个filter maxpooling之后的结果进行连接在一起，也就是按dim=1拼接，得到[batch_size, filter_num*len(kernel_list), 1, 1]
        out = out.view(in_size,-1) 
        # 转变为[batch_size, output_num]格式，output_num = filter * len(kernel_list)
        out = self.dropout(out)
        out = self.fc(out) 
        # 输出为：[batch_size, output_num]
        return out


def get_file_list(source_dir):
    file_list = []
    for root,dirs,files in os.walk(source_dir):
        # root: /Users/wuhanming/vscode_file/TextCNN/aclImdb/train
        # dirs: ['neg', 'pos']
        # files: ['.DS_Store']

        # root: /Users/wuhanming/vscode_file/TextCNN/aclImdb/train/neg
        # dirs: []
        # files: ['1821_4.txt', '10402_1.txt', '1062_4.txt', '9056_1.txt', …… ……

        # root: /Users/wuhanming/vscode_file/TextCNN/aclImdb/train/pos
        # dirs: []
        # files: ['4715_9.txt', '12390_8.txt', '8329_7.txt', '9063_8.txt', …… ……
        file=[os.path.join(root,filename) for filename in files]
        # /Users/wuhanming/vscode_file/TextCNN/aclImdb/train/neg/1821_4.txt
        file_list.extend(file)
    return file_list

def get_label_list(file_list):
    # 提取file_list的标签名
    label_name_list = [file.split('/')[-2] for file in file_list]
    label_list = []
    # 转换为数字表示 0--neg  1--pos  其他如DS_Store忽略不管
    for label_name in label_name_list:
        if label_name=="neg":
            label_list.append(0)
        elif label_name=="pos":
            label_list.append(1)
    return label_list

# 根据评论内容生成tensor。sentence是一个list，过滤停用词，得到sentence。embedding是词向量对象。word2id是字典。
# 注意：定义的神经网络是4维的：[batch_size, channel, max_size, embedding_dim]，channel均为1。
def generate_tensor(sentence, sentence_max_size, embedding, word2id):
    tensor=torch.zeros([sentence_max_size, embedding.embedding_dim])
    for index in range(sentence_max_size):
        # 在glove里面，用pad对应的vector进行补齐，而不是用zero
        if index>=len(sentence):
            tensor[index]=embedding.weight[word2id['pad']]
        # max_size大于实际长度。
        else:
            word = sentence[index]
            # 遍历，取第index个词。
            if word in word2id:
                vector = embedding.weight[word2id[word]]
                # 参考word2id字典的定义。word2id[word]映射到word在词向量模型中的标号id，weight[id]得到word的词向量。
                tensor[index]=vector
                # tensor的第index个word的词向量赋值。
            
            else:
                tensor[index]=embedding.weight[word2id['unk']]
    
    return tensor.unsqueeze(0) # 将tensor扩充为3维   [1, sentence_max_size, embedding_dim]

def load_stopwords(stopwords_dir):
    stopwords=[]
    with open(stopwords_dir,"r",encoding="utf-8") as file:
        for line in file.readlines():
            stopwords.append(line.strip())
    return stopwords

def segment(content, stopwords):
    res = []
    r='[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。“”！，]+'
    r1='br'
    
    for word in jieba.cut(content):
        word = word.replace(r1,"")
        word=re.sub(r,'',word)
        
        if word.strip()!="":
            res.append(word.lower().replace("isn't","is not").replace("won't","will not").replace("hasn't","has not").replace("wasn't","was not")\
                .replace("weren't","were not").replace("didn't","did not").replace("don't","do not").replace("couldn't","could not").replace("aren't","are not")\
                    .replace("can't","can not").replace("it's","it is").replace("i'm","i am").replace("you're","you are").replace("he's","he is").replace("she's","she is"))
    return res


def train_textcnn_model(net, train_loader, dev_loader, epoch, lr, weight_decay):
    print(datetime.datetime.now())
    print('开始训练---------------------')
    optimizer = optim.Adam(net.parameters(),lr=lr,weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    global_step=0
    best_acc = 0.0
    best_step = 0
    for j in range(epoch):
        for batch_idx, (data,target) in enumerate(train_loader):
            net.train()
            output = net(data.cuda())
            loss = criterion(output.cpu(),target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step+=1

            if((global_step)%50==0):
                print('train epoch: %d, global_step: %d, loss: %.6f'%(j,global_step,loss.item()/batch_size))

            if((global_step)%1000==0):
                net.eval()
                all_eval_logits=[]
                all_eval_labels=[]
                
                dev_acc = 0.0
                for i,(data,label) in enumerate(dev_loader):
                    output=net(data.cuda()).cpu()
                    logit=torch.argmax(output,dim=1)
                    all_eval_logits.extend(logit.tolist())
                    all_eval_labels.extend(label.tolist())
                    
                dev_acc = accuracy_score(all_eval_logits,all_eval_labels)
                print('epoch is :%d, global step is: %d, dev acc is: %6f'%(j,global_step,dev_acc))
                if(dev_acc > best_acc):
                    best_acc = dev_acc
                    torch.save(net,'best_model.bin')
                    best_step = global_step
                    print(datetime.datetime.now())
                    print('best acc has changed, best_acc is: ',best_acc)
    print("all training is finished, the best train step is %d, the accuracy is %6f"%{best_step, best_acc})

def textcnn_model_test(test_dataloader):
    print(datetime.datetime.now())
    print('开始测试：------------------------------------------------')
    model=torch.load('best_model.bin')
    model.eval()
    all_eval_logits = []
    for data in test_dataloader:
        output = model(data[0].cuda())
        logit = torch.argmax(output, dim=1)
        all_eval_logits.extend(logit.tolist())
    print(datetime.datetime.now())
    print(all_eval_logits)
    fout = open('result.txt','w',encoding='utf-8')
    results = ["neg\n" if res==0 else "pos\n" for res in all_eval_logits]
    fout.writelines(results)

#logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
# Python内置的标准模块，主要用于输出运行日志，%(asctime)s: 打印日志的时间， %(levelname)s: 打印日志级别名称，%(message)s: 打印日志信息。
train_dir="aclImdb/train"
dev_dir="aclImdb/test"
# 训练集 测试集路径
stopwords_dir="stopwords.txt"
word2vec_dir="/data/wuhanming/textcnn/glove.model.6B.300d.txt"
# 要用绝对路径

sentence_max_size=500 # 最大长度
batch_size=64
filter_num=512 # 每种卷积核数量
epoch=50
kernel_list=[2,3,4,5] # 卷积核大小
label_size=2
lr=1e-4
weight_decay=1e-3 # 1e-4

# 加载词向量模型
print(datetime.datetime.now())
print("加载词向量模型")
stopwords = load_stopwords(stopwords_dir)
# 加载词向量模型
wv = KeyedVectors.load_word2vec_format(word2vec_dir,binary=False)
'''
该结构称为KeyedVectors，实质上是实体和向量之间的映射。每个实体由其字符串id标识，因此是字符串和1维数组之间的映射关系。
实体通常对应一个单词，因此是将单词映射到一维向量。
'''
word2id={} 
# 存储词向量模型中 {word:idx} 的映射，也就是给词向量模型中每个词一个数字标记。
for i,word in enumerate(wv.index2word):
    word2id[word]=i
    
# 根据训练好的词向量模型，生成embedding对象。
embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors))
    
print(datetime.datetime.now())
print('获取数据')
'''
# 获取训练数据
train_set = get_file_list(train_dir)
#train_set=train_set[1:]
train_label = get_label_list(train_set)
print('train size:',len(train_set))
train_dataset = MyDataset(train_set,train_label,sentence_max_size,embedding,word2id,stopwords)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=3)

# 获取验证数据
dev_set = get_file_list(dev_dir)
dev_label = get_label_list(dev_set)
print('dev size:',len(dev_set))
dev_dataset = MyDataset(dev_set,dev_label,sentence_max_size,embedding,word2id,stopwords)
dev_dataloader = DataLoader(dev_dataset,batch_size=batch_size,shuffle=True,num_workers=3)
'''
# 获取测试数据
test_dir = "test.txt"
test_data = read_testdata(test_dir)
print('读取完毕')
dataset = data.TensorDataset(test_data)
test_dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)



is_train=False
if(is_train):
    # 定义模型
    net = TextCNN(vec_dim=embedding.embedding_dim,filter_num=filter_num,sentence_max_size=sentence_max_size,
                label_size=label_size,kernel_list=kernel_list).cuda()
    train_textcnn_model(net,train_dataloader,dev_dataloader,epoch,lr,weight_decay)
else:
    textcnn_model_test(test_dataloader)