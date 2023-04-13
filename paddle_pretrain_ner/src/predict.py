#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   predict.py
Author:   dangjinming(776039904@qq.com)
Date  :   2022/3/16
Desc  :   predict入口
"""

import os
import sys
import json
import logging
import configparser
import numpy as np
import paddle
from paddlenlp.transformers import AutoTokenizer,ErnieForTokenClassification

import dygraph
from data_loader import DataLoader
from label_encoder import LabelEncoder
from models.ernie_2fc import Ernie2FcForTokenClassification
from models.ernie_crf_v1 import ErnieCrfForTokenClassification
from models.ernie_crf_v2 import ErnieCrfV2ForTokenClassification


class Predict:
    """模型预测
    """
    def __init__(self, predict_conf_path):
        self.predict_conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.predict_conf.read(predict_conf_path)

        self.label_encoder = LabelEncoder(label_id_info=self.predict_conf["DATA"]["label_encoder_path"],
                                    isFile=True)
        for label_id, label_name in sorted(self.label_encoder.id_label_dict.items(), key=lambda x:x[0]):
            logging.info("%d: %s" % (label_id, label_name))

        self.tokenizer = AutoTokenizer.from_pretrained(self.predict_conf["ERNIE"]["pretrain_model"])

    def run(self):
        """执行入口
        """
        if self.predict_conf["RUN"].getboolean("finetune_ernie"):
            label_encoder = self.label_encoder
            predict_conf = self.predict_conf["ERNIE"]

            if self.predict_conf["ERNIE"].getboolean("use_crf"):
                if self.predict_conf["ERNIE"]["crf_type"] == "v1":
                    model_predict = ErnieCrfForTokenClassification.from_pretrained(self.predict_conf["ERNIE"]["pretrain_model"],
                                                                           num_classes=label_encoder.size() * 2 - 1)
                else:
                    ernie = ErnieForTokenClassification.from_pretrained(self.predict_conf["ERNIE"]["pretrain_model"],
                                                                        num_classes=label_encoder.size() * 2 - 1)
                    model_predict = ErnieCrfV2ForTokenClassification(ernie)

            else:
                model_predict = Ernie2FcForTokenClassification.from_pretrained(self.predict_conf["ERNIE"]["pretrain_model"],
                                                                       num_classes=label_encoder.size())

            dygraph.load_model(model_predict, self.predict_conf["MODEL_FILE"]["model_best_path"])

            predict_data = []
            text_list = []
            length_list = []
            origin_texts = []
            mark = 0
            tmp_d = {}
            for line in sys.stdin:
                mark = mark + 1
                cols = line.strip("\n").split("\t")
                origin_text = cols[0]
                text = self.tokenizer(" ".join(list(origin_text)))['input_ids']
                text_list.append(text)
                length_list.append(len(text))
                origin_texts.append(origin_text)
                tmp_d[origin_text] = cols

                if mark == 32:
                    predict_data = list(zip(text_list, length_list))
                    pre_label, pre_entity = dygraph.predict(model=model_predict,
                                                            predict_data=predict_data,
                                                            label_encoder=label_encoder,
                                                            batch_size=predict_conf.getint("batch_size"),
                                                            max_seq_len=predict_conf.getint("max_seq_len"),
                                                            max_ensure=True,
                                                            with_label=False,
                                                            use_crf=self.predict_conf["ERNIE"].getboolean("use_crf"))
                    for origin_text, entity in zip(origin_texts, pre_entity):
                        print(origin_text,"\t", "\t".join(tmp_d[origin_text][1:]), "\t", json.dumps(entity, ensure_ascii=False))
                    predict_data = []
                    text_list = []
                    length_list = []
                    origin_texts = []
                    mark = 0
            if mark != 0:
                predict_data = list(zip(text_list, length_list))
                pre_label, pre_entity = dygraph.predict(model=model_predict,
                                                        predict_data=predict_data,
                                                        label_encoder=label_encoder,
                                                        batch_size=predict_conf.getint("batch_size"),
                                                        max_seq_len=predict_conf.getint("max_seq_len"),
                                                        max_ensure=True,
                                                        with_label=False,
                                                        use_crf=self.predict_conf["ERNIE"].getboolean("use_crf"))
                for origin_text, entity in zip(origin_texts, pre_entity):
                    print(origin_text, "\t", "\t".join(tmp_d[origin_text][1:]), "\t", json.dumps(entity, ensure_ascii=False))

if __name__ == "__main__":
    Predict(predict_conf_path=sys.argv[1]).run()
