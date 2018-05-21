# encoding:utf-8
"""
description: this file helps to load raw file and gennerate batch x,y
author:luchi
date:22/11/2016
"""
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import codecs
import re
import copy


def dataMatrix(data, v_dimension, iteration):
    list_length = len(data)
    # vocabulary = []
    # for j in range(list_length):
    #     x_split = data[j].split(u' ')
    #     vocabulary.append(x_split)
    docs = []
    for j in range(list_length):
        data_j = []
        for i in data[j].split(u' '):
            data_j.append(i)
        docs.append(data_j)
    # docs = [data[j].split(u' ') for j in range(list_length)]
    print("word2vec training...")
    model = Word2Vec(docs, sg=0, window=7, min_count=1, size=v_dimension, workers=4, iter=iteration, sorted_vocab=None)
    return model


def save_to_test(data, path):
    writer = pd.ExcelWriter(path,
                            engine='xlsxwriter',
                            options={'strings_to_urls': False})
    data.to_excel(writer, sheet_name='Sheet1', index=None)  # 不要索引列
    writer.save()


def load_data_and_labels(cut_path, rule_path):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    cut_data = pd.read_excel(cut_path, 'Sheet1', encoding='gbk')
    rule_data = pd.read_excel(rule_path, 'Sheet1',  encoding='gbk')

    t_label = rule_data['category']
    t_num = rule_data['getNum']
    valid_df = []
    invalid_df = []
    for t in range(len(t_label)):
        df = pd.DataFrame(cut_data[cut_data['category'] == t_label[t]])
        df_len = np.arange(len(df)).tolist()
        np.random.shuffle(df_len)    # 随机打乱索引
        df['shuffle'] = df_len
        df = df.sort_values(by='shuffle')
        valid_df.append(df[:t_num[t]])
        invalid_df.append(df[t_num[t]:])
    result_set = pd.concat(valid_df, axis=0)  # 行连接
    result_set = pd.DataFrame(result_set)
    rs_len = np.arange(len(result_set)).tolist()
    np.random.shuffle(rs_len)
    result_set['shuffle'] = rs_len
    result_set = result_set.sort_values(by='shuffle')
    result_set.reset_index(drop=True, inplace=True)  # 重新设置行号

    remainder_set = pd.concat(invalid_df, axis=0)  # 行连接
    remainder_set = pd.DataFrame(remainder_set)
    re_len = np.arange(len(remainder_set)).tolist()
    np.random.shuffle(re_len)
    remainder_set['shuffle'] = re_len
    remainder_set = remainder_set.sort_values(by='shuffle')
    remainder_set.reset_index(drop=True, inplace=True)  # 重新设置行号

    label_set = []
    label_text = []
    label_dic = list({}.fromkeys(t_label).keys())
    for i in range(len(result_set)):
        each_label = [0] * len(label_dic)
        lid = label_dic.index(result_set['category'][i])
        each_label[lid] = 1
        label_set.append(each_label)
        label_text.append(result_set['category'][i])
    result_label = [np.array(label_set), np.array(label_text)]

    rlabel_set = []
    rlabel_text = []
    for i in range(len(remainder_set)):
        each_label = [0] * len(label_dic)
        lid = label_dic.index(remainder_set['category'][i])
        each_label[lid] = 1
        rlabel_set.append(each_label)
        rlabel_text.append(remainder_set['category'][i])
    remainder_label = [np.array(rlabel_set), np.array(rlabel_text)]

    return result_set, result_label, remainder_set, remainder_label


def load_data_and_labels_moyu(cut_path, rule_path):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    cut_data = pd.read_excel(cut_path, 'Sheet1', encoding='gbk')
    rule_data = pd.read_excel(rule_path, 'Sheet1',  encoding='gbk')

    t_label = rule_data['category']
    t_num = rule_data['getNum']
    valid_df = []
    for t in range(len(t_label)):
        df = pd.DataFrame(cut_data[cut_data['category'] == t_label[t]])
        df_len = np.arange(len(df)).tolist()
        np.random.shuffle(df_len)    # 随机打乱索引
        df['shuffle'] = df_len
        df = df.sort_values(by='shuffle')
        valid_df.append(df[:t_num[t]])
    result_set = pd.concat(valid_df, axis=0)  # 行连接
    result_set = pd.DataFrame(result_set)
    rs_len = np.arange(len(result_set)).tolist()
    np.random.shuffle(rs_len)
    result_set['shuffle'] = rs_len
    result_set = result_set.sort_values(by='shuffle')
    result_set.reset_index(drop=True, inplace=True)  # 重新设置行号

    label_set = []
    label_text = []
    label_dic = list({}.fromkeys(t_label).keys())
    for i in range(len(result_set)):
        lid = label_dic.index(result_set['category'][i])
        label_set.append(lid)
        label_text.append(result_set['category'][i])
    result_label = [np.array(label_set), np.array(label_text)]

    return result_set, result_label


def load_train_data(x, xp, y):
    train_set = (x, xp, y[:,  np.newaxis])
    return train_set


def load_predict_data(x, xp):
    test_set = (x, xp)
    return test_set


def clear_zero_in_list(x_list):
    while (True):
        last_xlist = x_list[-1]
        if last_xlist == 0:
            x_list.pop()
        else:
            break
    return len(x_list)


# return batch dataset
def batch_iter(data, batch_size, shuffle=True):      # data:  ([x],[y])
    # get dataset and label
    x, xp, y = data
    data_size = len(x)
    num_batches_per_epoch = int((data_size-1)/batch_size)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        x = x[shuffle_indices]
        xp = xp[shuffle_indices]
        y = y[shuffle_indices]
    for batch_index in range(num_batches_per_epoch):
        start_index = batch_index*batch_size
        end_index = min((batch_index+1)*batch_size, data_size)
        return_x = x[start_index:end_index]
        return_xp = xp[start_index:end_index]
        return_y = y[start_index:end_index]
        yield (return_x, return_xp, return_y)


def batch_iter_eval(data, batch_size, shuffle=False):  # data: []
    """
    Generates a batch iterator for a dataset.
    """
    x, xp = data
    data_size = len(x)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1   # int(9596-1)/64 +1 =150
    # Shuffle the data at each epoch
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        x_d = x[start_index:end_index]
        xp_d = xp[start_index:end_index]
        yield (x_d, xp_d)


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros
    ))
    return new_weights


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def load_train_sentences(path):
    """
    Load train sentences
    """
    data = pd.read_csv(path)
    context = data["Context"]
    contexts = [line for line in context]
    utterance = data["Utterance"]
    utterances = [line for line in utterance]
    label = list(data["Label"])
    y = np.zeros((len(label), 2), dtype=np.int64)
    for index, flag in enumerate(label):
        y[index][flag] = 1
    return contexts, utterances, np.array(label)


def load_predict_sentences(path):
    """
    Load test sentences
    """
    data = pd.read_csv(path)
    context = data["Context"]
    contexts = [line for line in context]
    utterance = data["Utterance"]
    utterances = [line for line in utterance]
    return contexts, utterances


def filter_predict_list(vocab_processor, x_o, xp_o):
    x_list = list(vocab_processor.transform(x_o))
    xp_list = list(vocab_processor.transform(xp_o))
    x_ch = list()
    xp_ch = list()
    newx_list = list()
    newxp_list = list()
    for x, xp, xo, xpo in zip(x_list, xp_list, x_o, xp_o):
        if x.any() and xp.any():
            x_ch.append(xo)
            xp_ch.append(xpo)
            newx_list.append(x)
            newxp_list.append(xp)
    return np.array(newx_list), np.array(newxp_list), np.array(x_ch), np.array(xp_ch)


def filter_train_list(x_list, xp_list, y_list):
    newx_list = list()
    newxp_list = list()
    newy_list = list()
    for x, xp, y in zip(x_list, xp_list, y_list):
        if x.any() and xp.any():
            newx_list.append(x)
            newxp_list.append(xp)
            newy_list.append(y)
    return np.array(newx_list), np.array(newxp_list), np.array(newy_list)