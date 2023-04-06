# -*- encoding:utf-8 -*-

import os
import random
import sys
import pickle
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch
from sklearn.preprocessing import LabelEncoder
from torch.optim import optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
import torch.nn as nn
import argparse
from multiprocessing import Process, Pool
import matplotlib.pyplot as plt
from sklearn import metrics
import torch.nn.functional as F
import sklearn
import pkuseg
from torch.autograd import Variable
from knowledgegraph import KnowledgeGraph


class DataPrecessForSingleSentence(object):
    """
    对文本进行处理
    """

    def __init__(self, bert_tokenizer, max_workers=10):
        self.bert_tokenizer = bert_tokenizer
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def get_input(self, sentences, max_seq_len=30):
        for i in range(len(sentences)):
            if type(sentences[i]) == float:
                sentences[i] = ''
        # 切词
        tokens_seq = list(
            self.pool.map(self.bert_tokenizer.tokenize, sentences))
        # 获取定长序列及其mask
        result = list(
            self.pool.map(self.trunate_and_pad, tokens_seq,
                          [max_seq_len] * len(tokens_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return seqs, seq_masks, seq_segments

    def get_input2(self, sentences, max_seq_len=30, max_seq_num=15):
        sentences_sep = []
        for each in sentences:
            len_each = len(each)
            if len_each < max_seq_num:
                for i in range(len_each, max_seq_num):
                    each.append('')
            else:
                each = each[:max_seq_num]
            sentences_sep.append(each)

        knowledges = []
        for sentences in sentences_sep:
            each_knowledge = []
            # 切词
            tokens_seq = list(
                self.pool.map(self.bert_tokenizer.tokenize, sentences))
            # 获取定长序列及其mask
            result = list(
                self.pool.map(self.trunate_and_pad, tokens_seq,
                            [max_seq_len] * len(tokens_seq)))
            for each in result:
                each_knowledge.append(list(each))
            knowledges.append(each_knowledge)

        return knowledges

    def trunate_and_pad(self, seq, max_seq_len):
        # 对超长序列进行截断
        if len(seq) > (max_seq_len - 2):
            seq = seq[0:(max_seq_len - 2)]
        # 分别在首尾拼接特殊符号
        seq = ['[CLS]'] + seq + ['[SEP]']
        # ID化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = [0] * len(seq) + padding
        # 对seq拼接填充序列
        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment

class CoAttention(nn.Module):
    def __init__(self, device, latent_dim = 200):
        super(CoAttention, self).__init__()
        
        self.linearq = nn.Linear(latent_dim, latent_dim)
        self.lineark = nn.Linear(latent_dim, latent_dim)
        self.linearv = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, sentence_rep, comment_rep, labels):
        
        query = self.linearq(sentence_rep)
        key = self.lineark(comment_rep)
        value = self.linearv(comment_rep)
        
        alpha_mat = torch.matmul(query, key.transpose(1,2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, value).squeeze(1)

        return x

class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class BertForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels=2, device='cuda'): # Change number of labels here.
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert_model')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.latent_dim = 768
        self.coattention = CoAttention(device, self.latent_dim)
        self.classifier = nn.Linear(self.latent_dim*2, num_labels)
        self.device = device
        nn.init.xavier_normal_(self.classifier.weight)
        self.tf_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=2)
        self.tf_encoder = nn.TransformerEncoder(self.tf_encoder_layer, num_layers=2)

    def forward_once(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)

        return pooled_output

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, labels=None):
        # forward pass of input 1
        output1 = self.forward_once(batch_seqs, batch_seq_masks, batch_seq_segments)
        output1 = torch.unsqueeze(output1, 1)
        # forward pass of input 2
        knowledges = batch_knowledges.cpu()
        knowledges = knowledges.numpy().tolist()
        tmp = []
        for each in knowledges:
            batch_seqs_k = [i[0] for i in each]
            batch_seq_masks_k = [i[1] for i in each]
            batch_seq_segments_k = [i[2] for i in each]
            t_batch_seqs_k = torch.tensor(batch_seqs_k, dtype=torch.long).to(self.device)
            t_batch_seq_masks_k = torch.tensor(batch_seq_masks_k, dtype = torch.long).to(self.device)
            t_batch_seq_segments_k = torch.tensor(batch_seq_segments_k, dtype = torch.long).to(self.device)
            output2 = self.forward_once(t_batch_seqs_k, t_batch_seq_masks_k, t_batch_seq_segments_k)
            tmp.append(output2)
        k_emb = torch.stack(tmp,dim=0)
        k_emb = self.tf_encoder(k_emb)
        k_emb = self.dropout(k_emb)
        pooled_output = self.coattention(output1, k_emb, labels)

        pooled_output = torch.cat([output1.squeeze(1), pooled_output], dim=1)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = FocalLoss(2)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--output_lossfig_path", default="./models/loss.png", type=str,
                        help="Path of the output model.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=10,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=6,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device use.")

    args = parser.parse_args()

    def set_seed(seed=7):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    set_seed(args.seed)
    
    # 读取数据
    train = pd.read_csv('data/train.tsv', encoding='utf-8', sep='\t')
    dev = pd.read_csv('data/dev.tsv', encoding='utf-8', sep='\t')
    test = pd.read_csv('data/test.tsv', encoding='utf-8', sep='\t')
    kg = ['CnDbpedia','webhealth','Medical']
    lookdown = False
    graph = KnowledgeGraph(kg, lookdown)

    # Load bert vocabulary and tokenizer
    bert_config = BertConfig('bert_model/bert_config.json')
    BERT_MODEL_PATH = 'bert_model'
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=False)

    # 产生输入数据
    processor = DataPrecessForSingleSentence(bert_tokenizer=bert_tokenizer)
    
    # train dataset
    seqs, seq_masks, seq_segments = processor.get_input(
        sentences=train['text_a'].tolist(), max_seq_len=args.seq_length)
    knowledges_triples = graph.get_knowledge(train['text_a'].tolist())
    knowledges = processor.get_input2(knowledges_triples)
    labels = train['label'].tolist()
    t_seqs = torch.tensor(seqs, dtype=torch.long)
    t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
    t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
    t_knowledges = torch.tensor(knowledges, dtype = torch.long)
    t_labels = torch.tensor(labels, dtype = torch.long)
    train_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_knowledges, t_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloder = DataLoader(dataset=train_data, sampler=train_sampler, batch_size=args.batch_size)

    # dev dataset
    seqs, seq_masks, seq_segments = processor.get_input(
        sentences=dev['text_a'].tolist(), max_seq_len=args.seq_length)
    knowledges_triples = graph.get_knowledge(dev['text_a'].tolist())
    knowledges = processor.get_input2(knowledges_triples)
    labels = dev['label'].tolist()
    t_seqs = torch.tensor(seqs, dtype=torch.long)
    t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
    t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
    t_knowledges = torch.tensor(knowledges, dtype = torch.long)
    t_labels = torch.tensor(labels, dtype = torch.long)
    dev_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_knowledges, t_labels)
    dev_sampler = RandomSampler(dev_data)
    dev_dataloder = DataLoader(dataset=dev_data, sampler=dev_sampler, batch_size=args.batch_size)

    # test dataset
    seqs, seq_masks, seq_segments = processor.get_input(
        sentences=test['text_a'].tolist(), max_seq_len=args.seq_length)
    knowledges_triples = graph.get_knowledge(test['text_a'].tolist(), './kg/test_triples.txt', True, test['label'].tolist())
    knowledges = processor.get_input2(knowledges_triples)
    labels = test['label'].tolist()
    t_seqs = torch.tensor(seqs, dtype=torch.long)
    t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
    t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
    t_knowledges = torch.tensor(knowledges, dtype = torch.long)
    t_labels = torch.tensor(labels, dtype = torch.long)
    test_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_knowledges, t_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloder = DataLoader(dataset=test_data, sampler=test_sampler, batch_size=args.batch_size)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build classification model
    model = BertForSequenceClassification(bert_config, 2, device)

    if device == 'cuda':
        if torch.cuda.device_count() > 1:
            print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
    model = model.to(device)

    # evaluation function
    def evaluate(args, is_test, metrics='f1'):
        if is_test:
            dataset = test_dataloder
            instances_num = test.shape[0]
            print("The number of evaluation instances: ", instances_num)
        else:
            dataset = dev_dataloder
            instances_num = dev.shape[0]
            print("The number of evaluation instances: ", instances_num)
        
        correct = 0
        model.eval()
        # Confusion matrix.
        confusion = torch.zeros(2, 2, dtype=torch.long)
        
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        for i, batch_data in enumerate(dataset):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, batch_labels = batch_data        
            with torch.no_grad():
                logits = model(
                    batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, labels=None)
            pred = logits.softmax(dim=1).argmax(dim=1)
            gold = batch_labels
            labels = batch_labels.data.cpu().numpy()
            predic = pred.data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            for j in range(pred.size()[0]):
                confusion[pred[j], gold[j]] += 1
            correct += torch.sum(pred == gold).item()
            
        if is_test:
            print("Confusion matrix:")
            print(confusion)
            print("Report precision, recall, and f1:")

        f1_avg = 0
        for i in range(confusion.size()[0]):
            if confusion[i,:].sum().item() == 0:
                p = 0
            else:
                p = confusion[i,i].item()/confusion[i,:].sum().item()
            if confusion[:,i].sum().item() == 0:
                r = 0
            else:
                r = confusion[i,i].item()/confusion[:,i].sum().item()
            if (p+r) == 0:
                f1 = 0
            else:
                f1 = 2*p*r / (p+r)
                f1_avg += f1
            if i == 1:
                label_1_f1 = f1
            print("Label {}: {:.4f}, {:.4f}, {:.4f}".format(i,p,r,f1))
        print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/instances_num, correct, instances_num))
        #print("labels_all", labels_all, predict_all, len(labels_all))
        test_auc = sklearn.metrics.roc_auc_score(labels_all, predict_all)
        print(test_auc)
        acc = sklearn.metrics.accuracy_score(labels_all, predict_all)
        report = sklearn.metrics.classification_report(labels_all, predict_all,digits=4)
        weighted_f1 = sklearn.metrics.f1_score(labels_all, predict_all, average='weighted')
        print(report)
        if metrics == 'Acc':
            return correct/instances_num
        elif metrics == 'f1':
            #return f1_avg/2
            return weighted_f1
        else:
            return correct/instances_num

    # training phase
    print("Start training.")
    instances_num = train.shape[0]
    batch_size = args.batch_size
    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)


    # 待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
            0.01
        },
        {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                        lr=args.learning_rate,
                        warmup=args.warmup,
                        t_total=train_steps)

    # 存储每一个batch的loss
    all_loss = []
    all_acc = []
    total_loss = 0.0
    result = 0.0
    best_result = 0.0

    for epoch in range(1, args.epochs_num+1):
        model.train()
        for step, batch_data in enumerate(train_dataloder):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, batch_labels = batch_data
            # 对标签进行onehot编码
            one_hot = torch.zeros(batch_labels.size(0), 2).long()
            '''one_hot_batch_labels = one_hot.scatter_(
                dim=1,
                index=torch.unsqueeze(batch_labels, dim=1),
                src=torch.ones(batch_labels.size(0), 2).long())

            
            logits = model(
                batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
            logits = logits.softmax(dim=1)
            loss_function = CrossEntropyLoss()
            loss = loss_function(logits, batch_labels)'''
            loss = model(
                batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, batch_labels)
            loss.backward()
            total_loss += loss.item()
            if (step + 1) % 100 == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.4f}".format(epoch, step+1, total_loss / 100))
                sys.stdout.flush()
                total_loss = 0.
            #print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, step+1, loss))
            optimizer.step()
            optimizer.zero_grad()
        
        all_loss.append(total_loss)
        total_loss = 0.
        print("Start evaluation on dev dataset.")
        result = evaluate(args, False)
        all_acc.append(result)
        if result > best_result:
            best_result = result
            #torch.save(model, open(args.output_model_path,"wb"))
            #save_model(model, args.output_model_path)
            torch.save(model.state_dict(), args.output_model_path)
        else:
            continue

        print("Start evaluation on test dataset.")
        evaluate(args, True)
    
    print('all_loss:', all_loss)
    print('all_acc:', all_acc)

    # Evaluation phase.
    print("Final evaluation on the test dataset.")
    model.load_state_dict(torch.load(args.output_model_path))
    evaluate(args, True)

    '''
    print(loss_collect)
    plt.figure(figsize=(12,8))
    plt.plot(range(len(loss_collect)), loss_collect,'g.')
    plt.grid(True)
    plt.savefig(args.output_lossfig_path)'''

if __name__ == "__main__":
    main()