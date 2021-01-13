import os
import logging

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchtext

def predict(iterator, model, args):
    model.eval()
    predictions = []
    for batch in iterator:
        feature = batch.text
        #print(feature)
        feature.t_()

        logit = model(feature.cuda())
        #logit = logit.cpu()
        for x in logit:
            if(x[0] < x[1]):
                predictions.append("1")
            else:
                predictions.append("0")
    with open("prediction.txt", "w", encoding="UTF-8") as fresult:
        for prediction in predictions:
            fresult.write(prediction+'\n')


def eval(iterator, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in iterator:
        feature, target = batch.text, batch.label
        feature.t_()
        target.sub_(1)

        logit = model(feature.cuda())
        loss = F.cross_entropy(logit.cpu(), target, size_average=False)
        avg_loss += loss.item()

        corrects += (torch.max(logit.cpu(), 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(iterator.dataset)
    avg_loss /= size
    accuracy = 100.0*corrects/size
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy

def train(train_iterator, validation_iterator, model, args):
    logging.info("Start training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history_best = -1.0
    idx = 1
    for epoch in range(1, args.epochs+1):
        print("Epoch : {}".format(epoch))
        for batch in train_iterator:
            model.train()
            feature, target = batch.text, batch.label
            feature.t_()
            target.sub_(1)
            
            logit = model(feature.cuda())
            loss = F.cross_entropy(logit.cpu(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logging.info("validation result :")
        accuracy = eval(validation_iterator, model, args)
        # 保存模型
        if(accuracy > history_best):
            torch.save(model, "./model/model_{}.pkl".format(idx))
            history_best = accuracy
            idx += 1
    logging.info("Finish")
