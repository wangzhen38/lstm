# -*- coding: utf-8 -*-
# @Time    : 18-8-9 下午1:39
# @Author  : WangZhen
# @FileName: out_word_vec.py
# @Software: PyCharm Community Edition


def init():
    dict = {}
    with open('word2vec/converted_word_Twitter.txt', encoding="utf-8") as f:
        for line in f:
            sentence = line.strip().split(' ')
            dict[sentence[0]] = sentence[1:]
    return dict


def word_v(str, dict):

    if str in dict:
        numbers = dict[str]
        numbers = [float(x) for x in numbers]
        return numbers
    sentence = []
    for i in range(300):
        sentence.append(0.0)
    return sentence


def add_word_v(train_data_voc_):
    dict = init()
    word_list = []
    for i in range(len(train_data_voc_)):
        lt = word_v(train_data_voc_[i], dict)
        word_list.append(lt)
    return word_list
