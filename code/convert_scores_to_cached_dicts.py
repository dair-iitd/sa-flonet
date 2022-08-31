import numpy as np
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import sys
import io
import os
import time
import copy
import argparse
from Model.ProxyScore import ProxyScore
import nltk
import pickle
import json
from shutil import copyfile
#from utils.SiameseData import SiameseData
from utils.Flowcharts import Flowcharts
from utils.ProxyScoreData import ProxyScoreData, ProxyScoreBatch
#from utils.proxy_scores import get_scores_dict
#from torch.utils.tensorboard import SummaryWriter
from utils import read_flowchart_jsons, read_dialog_jsons, build_vocab, create_batches, PAD, get_indexes_for_bleu, read_flowchart_doc_jsons, cache_embedding_matrix

from utils import PAD, PAD_INDEX, UNK, UNK_INDEX, GO_SYMBOL, GO_SYMBOL_INDEX, EOS, EOS_INDEX, EMPTY_INDEX, SEPARATOR, SEPARATOR_INDEX, CLS_INDEX
from utils import vectorize_text

import csv
import sklearn
from sklearn.model_selection import train_test_split

import logging
logger = logging.getLogger(__file__)


def save_dicts():


    # f1, f2, f3 contain only score json files generated from finetuned SemBERT
    # change the paths as appropriate

    f1 = open('new_cached_nli_scores/outdomain/only_circa/U-Flo_train_score_only.json','rb')
    train_scores = json.load(f1)
    f2 = open('new_cached_nli_scores/outdomain/only_circa/U-Flo_val_score_only.json','rb')
    val_scores = json.load(f2)
    f3 = open('new_cached_nli_scores/outdomain/only_circa/U-Flo_test_score_only.json','rb')
    test_scores = json.load(f3)

    # f4, f5, f6 contain the data that SemBERT evaluated scores on (the order of dialogs is the same in f1 and f4, f2 and f5, f3 and f6)

    f4 = open('new_cached_nli_scores/outdomain/Flonet_data/train.tsv')
    train_pairs = csv.reader(f4, delimiter='\t')
    f5 = open('new_cached_nli_scores/outdomain/Flonet_data/val.tsv')
    val_pairs = csv.reader(f5, delimiter='\t')
    f6 = open('new_cached_nli_scores/outdomain/Flonet_data/test.tsv')
    test_pairs = csv.reader(f6, delimiter='\t')

    # in the following, only the SemBERT scores for train data is cached
    # by making changes in the lines with comments val and test data can be cached as well

    train_pairs_list = []

    fields = []

    for i, row in enumerate(train_pairs):  # change to val_pairs or test_pairs as required
        if i == 0:
            fields = row
        else:
            train_pair_scored_entry = {}
            for j, value in enumerate(row):
                train_pair_scored_entry[fields[j]] = value 
            train_pair_scored_entry['logits'] = train_scores[i-1]['logits'] # change to val_scores or test_scores as required
            train_pairs_list.append(train_pair_scored_entry)

    train_data_cache_dict = {}

    names = flowcharts.get_flowchart_names()

    for name in names:
        train_data_cache_dict[name] = {}

    for i, entry in enumerate(train_pairs_list):
        fc = entry['flowchart']
        sentence1 = entry['sentence1']
        sentence2 = entry['sentence2']
        if sentence1 in train_data_cache_dict[fc].keys():
            train_data_cache_dict[fc][sentence1][sentence2] = entry['logits']
        else:
            train_data_cache_dict[fc][sentence1] = {}
            train_data_cache_dict[fc][sentence1][sentence2] = entry['logits']

    # change the name of the output json according to domain and split (train, val, test)

    with open('new_cached_nli_scores/outdomain/only_circa/U-Flo_train_score_cache.json',"w") as f: 
        json.dump(train_data_cache_dict,f, indent=4)


if __name__ == "__main__":
    save_dicts()