#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File  :   helper.py
Author:   dangjinming(776039904@qq.com)
Date  :   2022/3/16
Desc  :   common func
"""

import logging

# logging.basicConfig(format='%(asctime)s-%(levelname)s - %(message)s', level=logging.INFO)

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
                item.append([start - 1, end - 1, label_encoder.inverse_transform(start_type_id // 2 + 1)])
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
                item.append([start - 1, end - 1])
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
                        item.append([start - 1, end, label_encoder.inverse_transform(label[1][end])])
                        break
        pre_entity.append(item)
    return pre_entity






if __name__ == '__main__':
    print(extract_entity_crf([[1, 2, 2, 2, 1, 2, 2]], [7]))
