# -*- encoding:utf-8 -*-

import os
import random
import sys
import pickle
import pandas as pd
import numpy as np

class KnowledgeGraph(object):

    def __init__(self, spo_files, lookdown=False):
        self.lookdown = lookdown
        self.KGS = {'dsc': './kg/dsc.spo', 'CnDbpedia': './kg/CnDbpedia.spo', 'Medical': './kg/Medical.spo', 'webhealth': './kg/webhealth.spo'}
        self.spo_file_paths = [self.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        self.lookdown_table = self._create_lookdown_table()
        self.segment_vocab = list(self.lookup_table.keys())
        self.segment_vocab2 = list(self.lookdown_table.keys())
        self.vocab = self.segment_vocab.extend(self.segment_vocab2)
        self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.vocab)

    def _create_lookup_table(self):
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")    
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    value = pred + obje
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
        return lookup_table

    def _create_lookdown_table(self):
        lookdown_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")    
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    value = subj + pred
                    if obje in lookdown_table.keys():
                        lookdown_table[obje].add(value)
                    else:
                        lookdown_table[obje] = set([value])
        return lookdown_table

    def add_knowledge(self, sent, output_file, is_test):
        all_knowledge = []
        if is_test == True:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(sent+'\n')
        split_sent = self.tokenizer.cut(sent)
        if is_test == True:
            with open(output_file, 'a', encoding='utf-8') as f:
                for each in split_sent:
                    f.write(each+'\t')
                f.write('\n')
        know_sent = []
        for token in split_sent:
            entities = list(self.lookup_table.get(token, []))
            for each in entities:
                all_knowledge.append(token+each)
                if is_test == True:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(token)
                        f.write(each)
                        f.write('\n')
            if self.lookdown == True:
                entities = list(self.lookdown_table.get(token, []))
                for each in entities:
                    all_knowledge.append(each+token)
                    if is_test == True:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(each)
                            f.write(token)
                            f.write('\n')
        if is_test == True:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write('\n')
        
        return all_knowledge
    
    def get_knowledge(self, sentences, output_file='', is_test=False, label=None):
        knowledge = []
        for i in range(len(sentences)):
            if is_test == True:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(str(label[i]))
                    f.write(',')
            knowledge.append(self.add_knowledge(sentences[i], output_file, is_test))
        return knowledge