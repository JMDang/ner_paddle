#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   train.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   train入口
"""

import os
import sys
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

logging.basicConfig(
    format='"%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.INFO)

class Train:
    """模型训练
    """
    def __init__(self, train_conf_path):
        self.train_conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.train_conf.read(train_conf_path)

        Train.check_dir(self.train_conf["DEFAULT"]["input_dir"])
        Train.check_dir(self.train_conf["DEFAULT"]["output_dir"])
        Train.check_dir(self.train_conf["DEFAULT"]["model_dir"])

        self.label_encoder = LabelEncoder(label_id_info=self.train_conf["DATA"]["label_encoder_path"], 
                                    isFile=True)
        for label_id, label_name in sorted(self.label_encoder.id_label_dict.items(), key=lambda x:x[0]):
            logging.info("%d: %s" % (label_id, label_name))

        self.tokenizer = AutoTokenizer.from_pretrained(self.train_conf["ERNIE"]["pretrain_model"])
    
    def run(self):
        """执行入口
        """
        self.data_set = DataLoader(tokenizer=self.tokenizer, label_encoder=self.label_encoder)
        self.data_set.gen_data(
            train_data_dir=self.train_conf["DATA"]["train_data_path"],
            dev_data_dir=self.train_conf["DATA"]["dev_data_path"] if \
                self.train_conf["DATA"]["dev_data_path"] != "None" else None,
            test_data_dir=self.train_conf["DATA"]["test_data_path"] if \
                self.train_conf["DATA"]["test_data_path"] != "None" else None,
            use_crf=self.train_conf["ERNIE"].getboolean("use_crf")
        )

        if self.train_conf["RUN"].getboolean("finetune_ernie"):
            Train.finetune_ernie(self.train_conf,
                                 self.data_set,
                                 self.label_encoder)

        if self.train_conf["RUN"].getboolean("ernie_to_static"):
            self.ernie_to_static(train_conf=self.train_conf,
                                 label_encoder=self.label_encoder,
                                 )
            logging.info("[IMPORTANT]ernie model to static")
    
    @staticmethod
    def check_dir(dir_address):
        """检测目录是否存在
            1. 若不存在则新建
            2. 若存在但不是文件夹，则报错
            3. 若存在且是文件夹则返回
        """
        if not os.path.isdir(dir_address):
            if os.path.exists(dir_address):
                raise ValueError("specified address is not a directory: %s" % dir_address)
            else:
                logging.info("create directory: %s" % dir_address)
                os.makedirs(dir_address)
    
    @staticmethod
    def finetune_ernie(train_conf,
                       data_set,
                       label_encoder
                       ):
        """ernie微调
        """
        if train_conf["ERNIE"].getboolean("use_crf"):
            if train_conf["ERNIE"]["crf_type"] == "v1":
                model = ErnieCrfForTokenClassification.from_pretrained(train_conf["ERNIE"]["pretrain_model"],
                                                                   num_classes=label_encoder.size() *2 -1)
            else:
                ernie = ErnieForTokenClassification.from_pretrained(train_conf["ERNIE"]["pretrain_model"],
                                                                   num_classes=label_encoder.size() *2 -1)
                model = ErnieCrfV2ForTokenClassification(ernie)

        else:
            model = Ernie2FcForTokenClassification.from_pretrained(train_conf["ERNIE"]["pretrain_model"],
                                                                    num_classes=label_encoder.size())
        
        dygraph.load_model(model, train_conf["MODEL_FILE"]["model_best_path"] + ".pdparams")

        dygraph.train(model,
                      train_data=data_set.train_data,
                      label_encoder=label_encoder,
                      dev_data= data_set.dev_data,
                      epochs=train_conf["ERNIE"].getint("epoch"),
                      pretrain_lr=train_conf["ERNIE"].getfloat("learning_rate"),
                      other_lr=train_conf["ERNIE"].getfloat("other_learning_rate"),
                      weight_decay=train_conf["ERNIE"].getfloat("weight_decay"),
                      batch_size=train_conf["ERNIE"].getint("batch_size"),
                      max_seq_len=train_conf["ERNIE"].getint("max_seq_len"),
                      model_save_path=train_conf["MODEL_FILE"]["model_path"],
                      best_model_save_path=train_conf["MODEL_FILE"]["model_best_path"],
                      print_step=train_conf["ERNIE"].getint("print_step"),
                      use_crf=train_conf["ERNIE"].getboolean("use_crf"))
    
    @staticmethod
    def ernie_to_static(train_conf,
                        label_encoder):
        """ernie模型转静态图模型文件
        """
        if train_conf["ERNIE"].getboolean("use_crf"):
            if train_conf["ERNIE"]["crf_type"] == "v1":
                model = ErnieCrfForTokenClassification.from_pretrained(train_conf["ERNIE"]["pretrain_model"],
                                                                       num_classes=label_encoder.size() * 2 - 1)
            else:
                ernie = ErnieForTokenClassification.from_pretrained(train_conf["ERNIE"]["pretrain_model"],
                                                                    num_classes=label_encoder.size() * 2 - 1)
                model = ErnieCrfV2ForTokenClassification(ernie)

        else:
            model = Ernie2FcForTokenClassification.from_pretrained(train_conf["ERNIE"]["pretrain_model"],
                                                                   num_classes=label_encoder.size())
        dygraph.load_model(model, train_conf["MODEL_FILE"]["model_best_path"] + ".pdparams")

        model.eval()
        if train_conf["ERNIE"].getboolean("use_crf"):
            model = paddle.jit.to_static(model,
                                    input_spec=[paddle.static.InputSpec(shape=[None, train_conf["ERNIE"].getint("max_seq_len")],dtype="int32"),
                                                paddle.static.InputSpec(shape=[None,],dtype="int32"),
                                                None, None])
        else:
            model = paddle.jit.to_static(model,
                                         input_spec=[paddle.static.InputSpec(
                                          shape=[None, train_conf["ERNIE"].getint("max_seq_len")], dtype="int32"),
                                                 None, None, None])
        paddle.jit.save(model, train_conf["MODEL_FILE"]["model_static_path"])
        logging.info("save static ernie to {}".format(train_conf["MODEL_FILE"]["model_static_path"]))

if __name__ == "__main__":
    Train(train_conf_path=sys.argv[1]).run()



