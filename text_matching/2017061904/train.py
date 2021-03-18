import os

import torch
import torch.optim as optim
import torch.nn.functional as F

version = "v6.0"
log_dir = os.path.join("log", version)
model_dir = os.path.join("model", version)

def predict(iter, model, args):
    print("Predicting...")
    model.eval()

    predictions = []

    for batch in iter:
        sentence1 = batch.sentence1.t()
        sentence2 = batch.sentence2.t()

        res = model(sentence1.cuda(), sentence2.cuda())
        res = torch.max(res.cpu(), 1)[1]

        with open("data/test_result.txt", "a", encoding="UTF-8") as f:
            for ele in res:
                f.write(str(ele.item())+"\n")
        


def eval(iter, model, args):
    print("Evaluating...")
    model.eval()

    correct_num = 0
    total_num = 0

    for batch in iter:
        sentence1 = batch.sentence1.t()
        sentence2 = batch.sentence2.t()
        targets = batch.label.squeeze(0)
        
        if args.cuda:
            predictions = model(sentence1.cuda(), sentence2.cuda())
            predictions = torch.max(predictions.cpu(), 1)[1]
        else:
            predictions = model(sentence1, sentence2)
            predictions = torch.max(predictions, 1)[1]

        correct_num += (predictions == targets).sum()
        total_num += len(targets)
    
    accuracy = 100.0 * correct_num / total_num
    with open(os.path.join(log_dir, "acc.txt"), "a", encoding="UTF-8") as f:
        f.write("accuracy = {} / {} = {:.2f}%\n".format(correct_num, total_num, accuracy))

    return accuracy

def train(train_iter, valid_iter, model, args):
    #optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    history_best = 0.0
    for epoch in range(1, args.epoches+1):
        with open(os.path.join(log_dir, "acc.txt"), "a", encoding="UTF-8") as f:
            f.write("Epoch {}:\n".format(epoch))
        with open(os.path.join(log_dir, "loss.txt"), "a", encoding="UTF-8") as f:
            f.write("Epoch {}:\n".format(epoch))
        #print("Epoch : {}".format(epoch))
        # 训练
        model.train()
        batch_loss = 0.0
        for batch in train_iter:
            
            sentence1 = batch.sentence1.t()
            sentence2 = batch.sentence2.t()
            target = batch.label.squeeze(0)
            
            if(args.cuda):
                logit = model(sentence1.cuda(), sentence2.cuda())
                loss = F.cross_entropy(logit.cpu(), target)
            else:
                logit = model(sentence1, sentence2)
                loss = F.cross_entropy(logit, target)
            batch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with open(os.path.join(log_dir, "loss.txt"), "a", encoding="UTF-8") as f:
            f.write(str(batch_loss)+"\n")
        # 评估
        accuracy = eval(train_iter, model, args)
        accuracy = eval(valid_iter, model, args)
        if(accuracy > history_best):
            history_best = accuracy
            torch.save(model, os.path.join(model_dir, "model_{}.pkl".format(epoch)))
