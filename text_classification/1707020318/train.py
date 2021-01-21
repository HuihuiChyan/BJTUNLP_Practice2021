import os
import torch
from torch import nn
from torch import optim
from sklearn.metrics import accuracy_score
from torchtext import vocab as Vocab

import sys
sys.path.append('..')
from text_classify_v2 import pre_process as pre
from text_classify_v2 import textcnn

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
max_len = 500
embed_size = 300
kernel_sizes = [2, 3, 4, 5]
num_channels = [100, 100, 100, 100]
dropout_p = 0.5
epoch_num = 30
lr = 1e-4
weight_decay = 1e-7
is_train = False

train_data, valid_data = pre.file_read('train'), pre.file_read('test')

glove_vocab = Vocab.GloVe(name='6B', dim=300, cache=os.path.join(pre.data_root, 'glove'))
word2idx, idx2word = pre.build_vocab(glove_vocab)
# print('unk idx:', word2idx['unk'])  # 201534
# print('pad idx:', word2idx['pad'])  # 10109

train_iter = pre.get_loader(train_data, word2idx, batch_size, max_len)
valid_iter = pre.get_loader(valid_data, word2idx, batch_size, max_len)
print('data_iter loaded...')

def valid(model):
    model.eval()
    valid_acc, n = 0.0, 0
    with torch.no_grad():
        for X, y in valid_iter:
            X = X.to(device)
            y = y.to(device)
            output = model(X)

            y = y.data.cpu().numpy()
            output = output.argmax(dim=1).data.cpu().numpy()
            valid_acc += accuracy_score(y, output)
            n += 1
    return valid_acc / n
#train
if is_train:

    net = textcnn.TextCNN(len(word2idx), embed_size, kernel_sizes, num_channels, dropout_p)
    net.bilstm_embedding.weight.data.copy_(glove_vocab.vectors)
    net.const_embedding.weight.data.copy_(glove_vocab.vectors)
    net.const_embedding.requires_grad_(False)
    net.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()

    print('training on', device)
    best_acc = 0
    for epoch in range(epoch_num):
        net.train()
        train_loss, train_acc, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)

            output = net(X)
            l = loss(output, y).sum()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss += l / y.shape[0]
            train_acc += accuracy_score(y.data.cpu().numpy(), output.argmax(dim=1).data.cpu().numpy())
            n += 1

        valid_acc = valid(net)
        print('epoch = %d, loss = %.4f, train_acc = %.4f, test_acc = %.4f'
              % (epoch + 1, train_loss / n, train_acc / n, valid_acc))
        if valid_acc > best_acc and valid_acc > 0.89:
            best_acc = valid_acc
            torch.save(net, './model/model_(%.2f%%).pth' % (valid_acc * 100))
            print('model acc: %.2f%%, save...' % (valid_acc * 100))

# test
else:
    with open('./test_out_2.txt', 'w', encoding='utf-8') as f_write:
        test_iter = pre.get_test_data('./test.txt', word2idx, max_len, batch_size)
        print('test_iter loaded...')

        net = torch.load('./model/model_(90.34%).pth')
        print('model loaded...')
        net = net.to(device)

        valid_acc = valid(net)
        print('model acc: %.2f%%...' % (valid_acc * 100))

        net.eval()
        n = 0
        for X, y in test_iter:
            # print(type(X))
            X = X.to(device)
            output = net(X)
            predict = output.argmax(dim=1).data.cpu().numpy()
            for tag in predict:
                n += 1
                f_write.write(str('pos' if tag == 1 else 'net') + '\n')
            # predict = predict.squeeze(-1)
        print(n)
