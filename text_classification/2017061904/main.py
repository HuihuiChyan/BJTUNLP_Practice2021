import os
import model
import train
import argparse
import torch
import logging
import myDataset
import warnings
import torchtext.data as data
from torchtext.vocab import Vectors

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description="CNN text classifier")
# dataset
parser.add_argument("--batch-size", type=int, default=20)
# model
parser.add_argument("--kernel-sizes", type=str, default="3,4,5")
parser.add_argument("--dim", type=int, default=100)
parser.add_argument("--kernel-num", type=int, default=5)
# train
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=5)
# other
parser.add_argument("--train", action="store_true", default=False)

args = parser.parse_args()

def make_iterotor(text_field, label_field, args):
    print("Loading data...")
    train_data = myDataset.DS("./data/train", text_field, label_field, shuffle=True)
    valid_data = myDataset.DS("./data/test", text_field, label_field)
    print("Loading finished."+"\n")

    # cache = ".vector_cache"
    # if not os.path.exists(cache):
    #     os.mkdir(cache)
    # vectors = Vectors(name='./data/glove.42B.300d.txt', cache=cache)
    
    text_field.build_vocab(train_data, valid_data)
    """
    0 : <unk>
    1 : neg
    2 : pos
    """
    label_field.build_vocab(train_data, valid_data)

    test_data = myDataset.DS("./data/test.txt", text_field, label_field, test=True)
    #test_data = myDataset.DS("./features.txt", text_field, label_field, test=True)

    train_iter, valid_iter = data.Iterator.splits(
        (train_data, valid_data),
        batch_sizes = (args.batch_size, args.batch_size))
    test_iter = data.Iterator(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    return train_iter, valid_iter, test_iter

if __name__ == "__main__":
    # 建立Field
    text_field = data.Field(lower=True, fix_length=500)
    label_field = data.Field(sequential=False)
    train_iter, valid_iter, test_iter = make_iterotor(text_field, label_field, args)
    
    args.num = len(text_field.vocab)
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.class_num = len(label_field.vocab)-1

    if args.train is True:
        # 训练
        cnn = model.TextCNN(args)
        cnn.cuda()
        train.train(train_iter, valid_iter, cnn, args)
    else:
        # 预测
        print("Predicting...")
        cnn = torch.load("./model/model_{}.pkl".format(34))
        train.predict(test_iter, cnn.cuda(), args)
        print("Predicting finished."+"\n")
    
    
