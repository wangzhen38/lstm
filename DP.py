# -*- coding: utf-8 -*-
# @Time    : 18-8-6 上午11:01
# @Author  : WangZhen
# @FileName: DP.py
# @Software: PyCharm Community Edition
import re
import data_node


def datadeal(path=None, is_traindata=False):
    data_voc = {}
    with open(path, encoding='utf-8') as f:
        n_list = []
        for line in f:
            # 数据处理、分类
            dn = data_node.datanode()
            sentence, flag = line.strip().split('|||')
            flag = flag.strip()
            sentence = clean_str(sentence)
            sentence = sentence.strip()
            dn.train_data = sentence.split(' ')
            dn.train_data_word_num = len(dn.train_data)
            if flag == '2':
                continue
            if flag == '0':
                dn.train_data_label = 'negative'
            if flag == '1':
                dn.train_data_label = 'negative'
            if flag == '3':
                dn.train_data_label = 'positive'
            if flag == '4':
                dn.train_data_label = 'positive'
            n_list.append(dn)

            if is_traindata:
                # 建立词典
                sentence1 = sentence.split(' ')
                for i in range(len(sentence1)):
                    word = sentence1[i].lower()
                    if word in data_voc:
                        data_voc[word] += 1
                    else:
                        data_voc[word] = 1

    f.close()
    if is_traindata:
        v_list = []
        d_v = sorted(data_voc.items(), key=lambda x:x[1], reverse=True)
        v_list.append('<unk>')  # 可能会遇到新单词，不在此表内，统一按unk处理
        v_list.append('<pad>')  # 补句子长度专用
        for i in range(len(d_v)):
            v_list.append(d_v[i][0])
        return n_list, v_list
    else:
        return n_list


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
