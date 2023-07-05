#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File  :   data_loader.py
Author:   dangjinming(776039904@qq.com)
Date  :   2022/3/16
Desc  :   data_loader
"""

import os
import sys
import time
import json
import numpy as np
import logging
from collections import deque
logging.basicConfig(
    format='"%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.INFO)

class DataLoader:
    """数据集类
    """
    def __init__(self, tokenizer, label_encoder):
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
    
    def gen_data(self,
            train_data_dir=None,
            dev_data_dir=None,
            test_data_dir=None,
            use_crf=True):
        """加载训练集、验证集、测试集并进行序列化
        """
        if train_data_dir:
            self.train_text_list, self.train_text_ids_list, self.train_text_ids_length, \
                self.train_entity_list, self.train_label_list = \
                DataLoader.read_file(train_data_dir, self.tokenizer, self.label_encoder, use_crf)
            logging.info(f"train data num = {len(self.train_text_ids_list)}")
            self.train_data = list(zip(self.train_text_ids_list, self.train_text_ids_length, \
                                        self.train_label_list, self.train_entity_list))
        else:
            self.train_data = None

        if dev_data_dir:
            self.dev_text_list, self.dev_text_ids_list, self.dev_text_ids_length, \
                self.dev_entity_list, self.dev_label_list = \
                DataLoader.read_file(dev_data_dir, self.tokenizer, self.label_encoder, use_crf)
            logging.info("dev data num = {}".format(len(self.dev_text_ids_list)))
            self.dev_data = list(zip(self.dev_text_ids_list, self.dev_text_ids_length, \
                                        self.dev_label_list, self.dev_entity_list))
        else:
            self.dev_data = None
        
        if test_data_dir:
            self.test_text_list, self.test_text_ids_list, self.test_text_ids_length, \
            self.test_entity_list, self.test_label_list = \
                DataLoader.read_file(test_data_dir, self.tokenizer, self.label_encoder, use_crf)
            logging.info("test data num = {}".format(len(self.test_text_ids_list)))
            self.test_data = list(zip(self.test_text_ids_list, self.test_text_ids_length, \
                                     self.test_label_list, self.test_entity_list))
        else:
            self.test_data = None

    @staticmethod
    def read_file(data_dir, tokenizer, label_encoder, use_crf):
        """
        [IN] data_dir: 数据目录
             tokenizer: 切分工具
             label_encoder: 标签工具
             use_crf:使用crf进行实体识别,如果为false，则为双指针方式识别实体
        [OUT] res: list[]
        """
        text_list = []
        entity_list = []
        label_list = []
        file_list = DataLoader.get_file_list(data_dir)
        for file_index, file_path in enumerate(file_list):
            with open(file_path, "r", encoding='utf-8') as fr:
                for line in fr.readlines():
                    #先把有空格的数据去除掉，之前的实体index包含了空格
                    cols = line.strip("\n").split('\t')
                    if len(cols) != 2:#text \t labels
                        continue
                    text, label_entity_list = cols
                    if text.find(" ") != -1:
                        continue
                    if not label_entity_list:
                        continue
                    label_entity_list = json.loads(label_entity_list)
                    label_entity_list = [[item["start"], item["end"], item["label"]] for item in label_entity_list]
                    # label_entity_list = [[item["start"], item["end"]] for item in label_entity_list]
                    text_list.append(" ".join(list(text)))
                    entity_list.append(label_entity_list)
                    text_len = len(text) + 2
                    # crf标签序列
                    if use_crf:
                        """
                        class_name--->classify_id--->BIO_id
                        OTHER-------->0------------->OTHER:0
                        class1------->1------------->B_class1:1, I_class1:2
                        class2------->2------------->B_class2:3, I_class2:4 
                        class3------->3------------->B_class3:5, I_class3:6
                        ...
                        ...
                        class416----->416----------->B_class416:831, I_class3:832
                        
                        """
                        label = [0] * text_len
                        for item in label_entity_list:
                            start, end = item[:2]
                            if (end - 1) == start:
                                label[start+1] = label_encoder.transform(item[2]) * 2 - 1
                            else:
                                label[start + 1] = label_encoder.transform(item[2]) * 2 - 1
                                label[end] = label_encoder.transform(item[2]) * 2
                                label[start + 2: end] = [label_encoder.transform(item[2]) * 2] * (end - start - 2)
                    # 双指针预测,标签序列
                    else:
                        label = [[0] * text_len for _ in range(2)]
                        for item in label_entity_list:
                            label[0][item[0]+1] = label_encoder.transform(item[2])
                            label[1][item[1]] = label_encoder.transform(item[2])
                    label_list.append(label)
        logging.info("tokenizer encode start")
        start_time = time.time()
        text_ids_list = [tokenizer(x)['input_ids'] for x in text_list]
        logging.info(f"cost time:{time.time() - start_time: .4f}s")
        logging.info("tokenizer encode end")
        text_ids_length = [len(x) for x in text_ids_list]

        logging.info(":数据样例:")
        for i in range(3):
            logging.info(text_list[i])
            logging.info(json.dumps(text_ids_list[i]))
            logging.info(json.dumps(label_list[i]))
            logging.info(json.dumps(entity_list[i], ensure_ascii=False))
            logging.info("*"*77)
        return text_list, text_ids_list, text_ids_length, entity_list, label_list

    @staticmethod
    def get_file_list(data_path):
        """生成构成数据集的文件列表
        [in]  data_path : str, 数据集地址
        [out] file_list : list[str], 数据集文件名称列表
        """
        file_list = []
        path_stack = deque()
        path_stack.append(data_path)
        while len(path_stack) != 0:
            cur_path = path_stack.pop()
            if os.path.isdir(cur_path):
                files = os.listdir(cur_path)
                for file_name in files:
                    if not file_name or file_name.startswith("."):
                        continue
                    file_path = os.path.join(cur_path, file_name)
                    path_stack.append(file_path)
            elif os.path.isfile(cur_path):
                file_list.append(cur_path)
            else:
                raise TypeError(f"unknown type of data path : {cur_path}" )
        logging.info("文件样例:")
        for file in file_list[:5]:
            logging.info(f"filename:{file}")

        return file_list

    @staticmethod
    def batch_padding(data_list, max_seq_len=50, max_ensure=True):
        """ batch_padding
        data_list: list:(b,[m,n,s])
        max_seq_len: int, 数据最大长度
        max_ensure: bool, True则固定最大长度，否则按该批最大长度做padding(不超过max_seq_len的话)
        """
        if max_ensure:
            cur_max_len = max_seq_len
        else:
            cur_max_len = max([len(x) for x in data_list])
            cur_max_len = max_seq_len if cur_max_len > max_seq_len else cur_max_len
        # batch_padding
        res = [np.pad(x[:cur_max_len], [0, cur_max_len - len(x[:cur_max_len])], mode='constant') for x in data_list]
        return np.array(res)

    @staticmethod
    def batch_process(cur_batch_data, max_seq_len, max_ensure=True, with_label=True):
        """ 对批数据进行batch_padding处理
        [IN]  cur_batch_data:
                    ids_list:(b,[m,n,s...]),
                    length_list:(b,),
                    label_list:(b,2,[m,n,s...]) or (b,[m,n,s...])
        [OUT] batch_padding好的批数据
        """
        batch_list = []
        data_lists = list(zip(*cur_batch_data))

        ids_list = data_lists[0]
        #crf计算loss时会用到句子真实长度，避免出现越界错误
        length_list = [num if num <=max_seq_len else max_seq_len for num in data_lists[1]]
        ids_array = DataLoader.batch_padding(ids_list, max_seq_len, max_ensure)
        batch_list.append(ids_array)
        batch_list.append(length_list)

        if with_label:
            label_list = data_lists[2]
            entity_list = data_lists[3]
            label_shape = np.array(label_list).shape
            # shape(b,[m,n,s...])一致说明使用的是crf做ner;pad后shape(b,s)
            if np.shape(ids_list) == label_shape:
                label_array = DataLoader.batch_padding(label_list, max_seq_len, max_ensure)

            # 否则shape(b,2,[m,n,s...])是双指针ner，对于开始和结束位置都做一次分类;pad后shape(b,2,s)
            else:
                label_array = np.array(label_list)
                label_array = np.reshape(label_array, (label_shape[0] * label_shape[1],))
                label_array = DataLoader.batch_padding(label_array, max_seq_len, max_ensure)
                label_array = label_array.reshape((label_shape[0], label_shape[1], -1))
            batch_list.append(label_array)
            batch_list.append(entity_list)
        return batch_list

    @staticmethod
    def gen_batch_data(data_iter, batch_size=32, max_seq_len=50, max_ensure=True, with_label=True):
        """ 生成批数据
        [IN] data_iter: iterable, 可迭代的数据
            batch_size: int, 批大小
        """
        batch_data = list()
        for data in data_iter:
            if len(batch_data) == batch_size:
                yield DataLoader.batch_process(batch_data, max_seq_len, max_ensure, with_label)
                batch_data = list()
            batch_data.append(data)
        if len(batch_data) > 0:
            yield DataLoader.batch_process(batch_data, max_seq_len, max_ensure, with_label)

if __name__ == "__main__":

    from paddlenlp.transformers import AutoTokenizer
    from label_encoder import LabelEncoder
    tokenizer = AutoTokenizer.from_pretrained('ernie-1.0')

    label_encoder = LabelEncoder("../input/label.txt")
    dl = DataLoader(tokenizer, label_encoder)

    dl.gen_data("../input/train_data/", use_crf=True)

    train_data_batch = DataLoader.gen_batch_data(dl.train_data,
                                                 batch_size=10,
                                                 max_seq_len=64,
                                                 max_ensure=True
                                                 )

    for cur_train_data, cur_train_length, cur_train_label, cur_train_entity in train_data_batch:
        print("djm:", cur_train_data.shape)
        print("djm:", cur_train_data[0])
        print("djm:", cur_train_label.shape)
        print("djm:", cur_train_label[0])
        print("djm:", cur_train_label[7])
        print("djm:", cur_train_entity)
        print("djm:", cur_train_length)
        print(np.shape(cur_train_entity))
        break
