#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File  :   helper.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   common func
"""
from collections import defaultdict

def load_dict(dict_path):
    vocab = {}
    i = 0
    with open(dict_path, "r", encoding="utf-8") as fin:
        for line in fin:
            key = line.strip("\n")
            vocab[key] = i
            i += 1
    return vocab

def build_vocab(texts,
                vocab_save_path="./vocab.txt",
                stopwords=[],
                num_words=None,
                min_freq=10,
                pad_token="[PAD]",
                unk_token="[UNK]",
                num_token="[NUM]"):

    word_counts = defaultdict(int)
    for text in texts:
        if not text:
            continue
        for word in text.replace(' ', ''):
            if word in stopwords:
                continue
            if word.isdigit():
                word_counts["[NUM]"] += 1
            else:
                word_counts[word] += 1
    wcounts = []
    for word, count in word_counts.items():
        if count < min_freq:
            continue
        wcounts.append((word, count))
    wcounts.sort(key=lambda x: x[1], reverse=True)
    # -2 for the pad_token and unk_token which will be added to vocab.
    if num_words is not None and len(wcounts) > (num_words - 2):
        wcounts = wcounts[: (num_words - 2)]
    # add the special pad_token and unk_token to the vocabulary
    sorted_voc = [pad_token, unk_token, num_token]
    sorted_voc.extend(wc[0] for wc in wcounts)
    with open(vocab_save_path, "w", encoding="utf-8") as fw:
        for word in sorted_voc:
            fw.write(word + "\n")
    return True

def extract_entity_crf(pre_label, predict_length, label_encoder=None):
    """ 从预测标签抽取实体,标签个数: label_encoder.size() * 2 - 1
    [IN]  pre_label: 模型预测结果
          predict_length: 预测数据真实长度
          label_encoder: 标签编码
    """
    pre_entity = []
    for label, length in zip(pre_label, predict_length):
        item = []
        if length >= len(label):
            length = len(label)
        index = 0
        while index < length:
            if label[index] != 0:
                start = index
                start_type_id = label[index]
                end = index + 1
                while end < length and label[end]==start_type_id + 1:
                    end += 1
                    index += 1
                if start_type_id // 2 + 1 not in label_encoder.id_label_dict:
                    break
                item.append([start, end, label_encoder.inverse_transform(start_type_id // 2 + 1)])
            index += 1
        pre_entity.append(item)
    return pre_entity

def extract_entity_crf_BIO(pre_label, predict_length):
    """ 从预测标签抽取实体,只有三种标签   BIO
    [IN]  pre_label: 模型预测结果
          predict_length: 预测数据真实长度
          label_encoder: 标签编码
    """
    pre_entity = []
    for label, length in zip(pre_label, predict_length):
        item = []
        if length >= len(label):
            length = len(label)
        index = 0
        while index < length:
            if label[index] == 1:
                start = index
                end = index + 1
                while end < length and label[end]==2:
                    end += 1
                item.append([start, end])
            index += 1
        pre_entity.append(item)
    return pre_entity

def extract_entity_dp(pre_label, predict_length, label_encoder):
    """ 从预测标签抽取实体
    [IN]  pre_label: 模型预测结果
          predict_length: 预测数据真实长度
          label_encoder: 标签编码
    """
    pre_entity = []
    for label, length in zip(pre_label, predict_length):
        item = []
        if length >= len(label[0]):
            length = len(label[0])
        for start in range(length):
            if (label[0][start] != 0):
                for end in range(start, length):
                    if (label[1][end] != 0):
                        item.append([start, end + 1, label_encoder.inverse_transform(label[1][end])])
                        break
        pre_entity.append(item)
    return pre_entity






if __name__ == '__main__':
    print(extract_entity_crf([[1, 2, 2, 2, 1, 2, 2]], [7]))
